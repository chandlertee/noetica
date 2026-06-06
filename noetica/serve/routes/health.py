"""Health endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from noetica.serve.clients.ollama import OllamaClient, OllamaError
from noetica.serve.config import Settings, get_settings
from noetica.serve.models import Health

router = APIRouter(prefix="/v1", tags=["health"])


@router.get("/health", response_model=Health)
async def health(settings: Settings = Depends(get_settings)):
    expected = [settings.model_text, settings.model_vision, settings.model_embed]
    try:
        async with OllamaClient(settings.ollama_url, timeout=5.0) as client:
            models = await client.list_models()
    except OllamaError as e:
        return Health(
            ok=False,
            ollama_reachable=False,
            models_present={m: False for m in expected},
            detail=str(e),
        )

    # Ollama tags use names like "qwen2.5:7b-instruct"; allow loose match by tag prefix.
    have = set(models)
    bare = {name.split(":")[0] for name in models}
    present = {m: (m in have or m.split(":")[0] in bare) for m in expected}
    ok = all(present.values())
    return Health(
        ok=ok,
        ollama_reachable=True,
        models_present=present,
        detail=None if ok else "one or more expected models are not pulled",
    )
