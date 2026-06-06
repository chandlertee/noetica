"""Request / response schemas for the public API.

Field naming convention: snake_case throughout. The HTTP boundary always emits
strict JSON validated against these schemas — any field the model returns that
isn't here is silently dropped on the way out.

This service is domain-free: it ships only the generic structured-output,
embed, and health schemas. Domain-specific request/response shapes (book
metadata, etc.) live in the calling projects.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ----- shared -----


class Health(BaseModel):
    ok: bool
    ollama_reachable: bool
    models_present: dict[str, bool]
    detail: str | None = None


# ----- embed -----


class EmbedRequest(BaseModel):
    texts: list[str]
    model: str | None = Field(default=None, description="Override the configured embed model.")

    @field_validator("texts")
    @classmethod
    def _trim_texts(cls, v: list[str]) -> list[str]:
        return [t[:32_000] for t in v]


class EmbedResult(BaseModel):
    embeddings: list[list[float]]
    model: str
    dim: int


# ----- generic structured primitive -----


class StructuredRequest(BaseModel):
    """Generic JSON-Schema-driven generation request.

    Powers /v1/llm/structured: callers describe what they want with a JSON
    Schema (Draft 2020-12), supply a prompt, and we hand back validated JSON.
    """

    model: str | None = Field(
        default=None,
        description="Ollama model name. Defaults to MODEL_TEXT from server config.",
    )
    prompt: str
    response_schema: dict = Field(
        description="JSON Schema (Draft 2020-12) the output must conform to.",
    )
    images: list[str] = Field(
        default_factory=list,
        description="Base64-encoded images. Pair with a vision model name.",
    )
    options: dict = Field(
        default_factory=lambda: {"temperature": 0.0},
        description="Ollama generation options (temperature, top_p, num_ctx, ...).",
    )
    cache: bool = Field(
        default=True,
        description="Whether to consult/populate the disk cache.",
    )
    max_attempts: int = Field(default=2, ge=1, le=4)


class StructuredResponse(BaseModel):
    data: dict | list
    model: str
    cached: bool = False


# ----- error -----


class ApiError(BaseModel):
    error: str
    detail: str | None = None
    kind: Literal["validation", "upstream", "schema", "timeout", "unknown"] = "unknown"
