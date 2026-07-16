"""Generic LLM primitive.

`/v1/llm/structured` is the foundational (and only) generation endpoint: given
a prompt and a JSON Schema, return validated JSON.

This service is domain-free by design. Callers that want structured outputs
for a specific domain (movies, music, anything else) own their prompt
templates and response schemas on their side and drive them through this
endpoint — the service itself knows nothing about any domain.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from noetica.serve import cache as cache_mod
from noetica.serve.clients.ollama import (
    OllamaClient,
    OllamaSchemaError,
    OllamaTimeout,
    OllamaUnavailable,
    with_retry,
)
from noetica.serve.config import Settings, get_settings
from noetica.serve.deps import get_ollama
from noetica.serve.models import StructuredRequest, StructuredResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/llm", tags=["llm"])


@router.post("/structured", response_model=StructuredResponse)
async def structured(
    req: StructuredRequest,
    client: OllamaClient = Depends(get_ollama),
    settings: Settings = Depends(get_settings),
):
    """Generate JSON output validated against the supplied JSON Schema.

    On schema-validation failure we make one repair attempt (showing the
    model its previous output + the error) before returning 502.
    """
    model = req.model or settings.model_text

    # Cache key includes the schema so callers can't get a stale response
    # when they tighten their schema.
    cache_payload = {
        "kind": "llm_structured",
        "prompt": req.prompt,
        "schema": req.response_schema,
        "images_sig": [img[:64] for img in req.images],
        "options": req.options,
    }
    if req.cache and settings.cache_enabled:
        cached = cache_mod.get(settings.cache_dir, model, settings.prompt_version, cache_payload)
        if cached is not None:
            return StructuredResponse(data=cached, model=model, cached=True)

    try:
        # Retry only a cold/unreachable Ollama (connection refused); schema
        # repair is handled inside generate_with_json_schema, and timeouts
        # are surfaced as 504 rather than re-run.
        data = await with_retry(
            lambda: client.generate_with_json_schema(
                model,
                req.prompt,
                req.response_schema,
                images=req.images or None,
                options=req.options,
                max_attempts=req.max_attempts,
            ),
            retries=settings.ollama_max_retries,
            base_delay=settings.ollama_retry_base_delay,
        )
    except OllamaUnavailable as e:
        raise HTTPException(status_code=503, detail=f"ollama unavailable: {e}") from e
    except OllamaTimeout as e:
        raise HTTPException(status_code=504, detail=f"ollama timeout: {e}") from e
    except OllamaSchemaError as e:
        raise HTTPException(status_code=502, detail=f"schema validation: {e}") from e

    if req.cache and settings.cache_enabled:
        cache_mod.put(settings.cache_dir, model, settings.prompt_version, cache_payload, data)

    return StructuredResponse(data=data, model=model, cached=False)
