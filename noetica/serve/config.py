"""Runtime configuration loaded from environment variables.

Models are env-driven so the same image runs on a laptop (smaller models) and on
a workstation with a 24 GB GPU (larger models) without code changes.

Env prefix is ``NOETICA_`` for service settings; the Ollama runtime and model
selectors keep their conventional unprefixed names (``OLLAMA_URL``,
``MODEL_TEXT``, ...) so they read the same here as in the Ollama docs.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Ollama
    ollama_url: str = Field(default="http://localhost:11434", alias="OLLAMA_URL")
    # Vision models on CPU-only laptops are slow — minutes per image is normal.
    # Set a generous default so cold-start vision calls don't time out the worker.
    # Override per-deploy: a GPU host can drop this back to 60s.
    ollama_timeout_seconds: float = Field(default=600.0, alias="OLLAMA_TIMEOUT_SECONDS")

    # Transient-failure retry. A cold or just-restarted Ollama refuses
    # connections for a moment; rather than 503 on the first failure we retry
    # the connection with exponential backoff (delay = base_delay * 2**attempt).
    # Only connection failures are retried — a slow generation still fails at
    # the timeout above. Set retries to 0 to disable.
    ollama_max_retries: int = Field(default=2, alias="OLLAMA_MAX_RETRIES")
    ollama_retry_base_delay: float = Field(default=0.5, alias="OLLAMA_RETRY_BASE_DELAY")

    # Models
    model_text: str = Field(default="qwen2.5:7b-instruct", alias="MODEL_TEXT")
    # The Ollama tag is `qwen2.5vl` (no hyphen). Only invoked when a caller
    # passes images on /v1/llm/structured.
    model_vision: str = Field(default="qwen2.5vl:7b", alias="MODEL_VISION")
    model_embed: str = Field(default="nomic-embed-text", alias="MODEL_EMBED")

    # Service
    host: str = Field(default="127.0.0.1", alias="NOETICA_HOST")
    port: int = Field(default=8001, alias="NOETICA_PORT")

    # Structured JSON logs to stdout. Flip off for human-readable logs in dev.
    json_logs: bool = Field(default=True, alias="NOETICA_JSON_LOGS")
    log_level: str = Field(default="INFO", alias="NOETICA_LOG_LEVEL")

    # Cache
    cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "noetica",
        alias="NOETICA_CACHE_DIR",
    )
    cache_enabled: bool = Field(default=True, alias="NOETICA_CACHE_ENABLED")

    # Cache namespace — bump to invalidate the disk cache wholesale (e.g. after a
    # model upgrade changes outputs). Callers own their prompts, so this is just
    # a coarse cache-busting knob.
    prompt_version: str = Field(default="v3", alias="NOETICA_PROMPT_VERSION")

    # Security / access control
    #
    # `api_key`: when set, every /v1/* request must include the matching value
    # in the `X-API-Key` header (or `Authorization: Bearer <key>`). Empty
    # string (the default) disables the check — safe for localhost-only use,
    # MUST be set before exposing on the LAN.
    api_key: str = Field(default="", alias="NOETICA_API_KEY")

    # `cors_origins`: comma-separated list of origins allowed to call the API
    # from a browser. Defaults to the Open WebUI dev origin and localhost. Use
    # "*" to allow any (NOT recommended once api_key is set, since the CORS
    # wildcard disables credentials).
    cors_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000,http://localhost:8080",
        alias="NOETICA_CORS_ORIGINS",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.cache_dir.mkdir(parents=True, exist_ok=True)
    return s
