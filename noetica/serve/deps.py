"""Shared FastAPI dependencies."""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Depends

from noetica.serve.clients.ollama import OllamaClient
from noetica.serve.config import Settings, get_settings


async def get_ollama(
    settings: Settings = Depends(get_settings),
) -> AsyncIterator[OllamaClient]:
    """Yield a per-request Ollama client, closed when the request finishes."""
    async with OllamaClient(settings.ollama_url, timeout=settings.ollama_timeout_seconds) as c:
        yield c
