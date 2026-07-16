"""Embedding endpoint — passthrough to Ollama's /api/embed."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from noetica.serve.clients.ollama import (
    OllamaClient,
    OllamaTimeout,
    OllamaUnavailable,
    with_retry,
)
from noetica.serve.config import Settings, get_settings
from noetica.serve.deps import get_ollama
from noetica.serve.models import EmbedRequest, EmbedResult

router = APIRouter(prefix="/v1", tags=["embed"])


@router.post("/embed", response_model=EmbedResult)
async def embed(
    req: EmbedRequest,
    client: OllamaClient = Depends(get_ollama),
    settings: Settings = Depends(get_settings),
):
    model = req.model or settings.model_embed
    if not req.texts:
        return EmbedResult(embeddings=[], model=model, dim=0)
    try:
        # Retry only a cold/unreachable Ollama; timeouts surface as 504.
        embeddings = await with_retry(
            lambda: client.embed(model, req.texts),
            retries=settings.ollama_max_retries,
            base_delay=settings.ollama_retry_base_delay,
        )
    except OllamaUnavailable as e:
        raise HTTPException(status_code=503, detail=f"ollama unavailable: {e}") from e
    except OllamaTimeout as e:
        raise HTTPException(status_code=504, detail=f"ollama timeout: {e}") from e
    dim = len(embeddings[0]) if embeddings else 0
    return EmbedResult(embeddings=embeddings, model=model, dim=dim)
