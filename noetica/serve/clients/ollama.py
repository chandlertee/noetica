"""Async HTTP client for Ollama.

We use Ollama's `/api/generate` with `format: "json"` so the model is forced
into valid JSON. We then validate against a caller-supplied JSON Schema.
If validation fails, we ask the model once more to repair its output before
giving up — this is cheaper and more reliable than tolerating loose JSON.

We deliberately do not use Ollama's `/api/chat`. Generate is stateless and the
extra structure of chat messages adds nothing for one-shot extraction tasks.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Base for Ollama client errors."""


class OllamaUnavailable(OllamaError):
    """Couldn't reach Ollama at all."""


class OllamaTimeout(OllamaError):
    """Ollama took too long."""


class OllamaSchemaError(OllamaError):
    """Ollama returned JSON that didn't match the requested schema, even after a repair pass."""


class OllamaClient:
    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> OllamaClient:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self._timeout, connect=5.0),
        )
        return self

    async def __aexit__(self, *exc):
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("OllamaClient must be used as an async context manager")
        return self._client

    # ----- low level -----

    async def list_models(self) -> list[str]:
        try:
            resp = await self.client.get("/api/tags")
            resp.raise_for_status()
        except httpx.ConnectError as e:
            raise OllamaUnavailable(f"connect failed: {e}") from e
        except httpx.HTTPStatusError as e:
            raise OllamaError(f"/api/tags returned {e.response.status_code}") from e
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]

    async def is_reachable(self) -> bool:
        try:
            await self.list_models()
            return True
        except OllamaError:
            return False

    async def _generate_raw(
        self,
        model: str,
        prompt: str,
        *,
        images: list[str] | None = None,
        format_json: bool = True,
        ollama_format: Any = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        """Call /api/generate with non-streaming response. Returns the raw text.

        `ollama_format`: if provided, sent as the `format` field (Ollama
        accepts a JSON Schema directly here in modern versions, falling back
        to plain JSON mode if the model doesn't honor it). Overrides
        `format_json`.
        """
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options or {"temperature": 0.1},
        }
        if ollama_format is not None:
            body["format"] = ollama_format
        elif format_json:
            body["format"] = "json"
        if images:
            body["images"] = images

        try:
            resp = await self.client.post("/api/generate", json=body)
        except httpx.ConnectError as e:
            raise OllamaUnavailable(f"connect failed: {e}") from e
        except httpx.TimeoutException as e:
            raise OllamaTimeout(f"generate timed out after {self._timeout}s") from e

        if resp.status_code >= 400:
            raise OllamaError(f"/api/generate returned {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        return data.get("response", "")

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via /api/embed (the modern batch endpoint)."""
        body = {"model": model, "input": texts}
        try:
            resp = await self.client.post("/api/embed", json=body)
        except httpx.ConnectError as e:
            raise OllamaUnavailable(f"connect failed: {e}") from e
        except httpx.TimeoutException as e:
            raise OllamaTimeout(f"embed timed out after {self._timeout}s") from e
        if resp.status_code >= 400:
            raise OllamaError(f"/api/embed returned {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        return data.get("embeddings", [])

    # ----- structured -----

    async def generate_with_json_schema(
        self,
        model: str,
        prompt: str,
        json_schema: dict[str, Any],
        *,
        images: list[str] | None = None,
        options: dict[str, Any] | None = None,
        max_attempts: int = 2,
    ) -> dict[str, Any]:
        """Generate JSON output validated against a *JSON Schema* dict.

        This is the primitive that powers /v1/llm/structured: callers describe
        what they want with a JSON Schema and get back validated JSON.

        We pass the schema as Ollama's `format` parameter when supported
        (recent versions accept a JSON Schema directly there). We *also*
        run jsonschema validation on our side, both to provide a clear
        error and because the model may still drift on older Ollama builds.
        """
        try:
            from jsonschema import Draft202012Validator
        except ImportError as e:  # pragma: no cover
            raise OllamaError("jsonschema package is required for /v1/llm/structured") from e

        validator = Draft202012Validator(json_schema)

        def validate(raw: str) -> dict[str, Any]:
            data = json.loads(raw)
            errors = list(validator.iter_errors(data))
            if errors:
                # Surface the first 3 errors — that's plenty for the repair pass.
                first = "; ".join(
                    f"{'.'.join(map(str, e.absolute_path))}: {e.message}" for e in errors[:3]
                )
                raise ValueError(f"schema: {first}")
            return data

        return await self._structured_loop(
            model,
            prompt,
            validate,
            images=images,
            options=options,
            max_attempts=max_attempts,
            ollama_format=json_schema,
        )

    async def _structured_loop(
        self,
        model: str,
        prompt: str,
        validate,
        *,
        images: list[str] | None = None,
        options: dict[str, Any] | None = None,
        max_attempts: int = 2,
        ollama_format: Any = None,
    ):
        """Retry/repair driver for the structured-output path.

        `validate` parses+checks the raw model output and either returns the
        validated value or raises ValueError (json.JSONDecodeError is a
        ValueError subclass) describing what was wrong, which we feed back to
        the model on the repair attempt.
        """
        last_raw = ""
        last_err: str | None = None
        for attempt in range(max_attempts):
            current_prompt = prompt
            if attempt > 0 and last_err:
                current_prompt = (
                    f"{prompt}\n\n"
                    f"Your previous response was:\n{last_raw}\n\n"
                    f"That failed validation with: {last_err}\n"
                    f"Return ONLY valid JSON matching the requested schema."
                )
            raw = await self._generate_raw(
                model,
                current_prompt,
                images=images,
                options=options,
                # ollama_format=None means use plain "json" mode; otherwise pass schema.
                format_json=ollama_format is None,
                ollama_format=ollama_format,
            )
            last_raw = raw
            try:
                return validate(raw)
            except ValueError as e:
                # Bad JSON or a schema mismatch; both surface as ValueError.
                last_err = str(e)
                logger.warning(
                    "ollama output failed validation (attempt %d): %s", attempt + 1, last_err
                )
                continue
        raise OllamaSchemaError(f"after {max_attempts} attempts: {last_err}; last={last_raw[:400]}")


async def with_retry(
    coro_fn,
    *,
    retries: int = 2,
    base_delay: float = 1.0,
):
    """Run an async callable with exponential backoff on OllamaUnavailable.

    Only connection failures are retried — a cold or just-started Ollama
    refuses connections for a moment, and a short backoff lets it come up
    rather than failing the request with an immediate 503.

    Timeouts are deliberately *not* retried: a slow generation should surface
    as a 504 rather than be re-run, which would only multiply an already-long
    wait. Pass ``retries=0`` to disable retrying entirely.
    """
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return await coro_fn()
        except OllamaUnavailable as e:
            last_exc = e
            if attempt >= retries:
                break
            delay = base_delay * (2**attempt)
            logger.warning("ollama unavailable (%s); retrying in %.1fs", e, delay)
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc
