"""FastAPI app entry point.

Wires up, from the outside in:
  · request-context middleware — assigns an ``X-Request-ID`` and emits a
    structured access log line per request.
  · CORS — env-configured origin list (``NOETICA_CORS_ORIGINS``).
  · API-key middleware — no-op unless ``NOETICA_API_KEY`` is set.
  · metrics middleware — records Prometheus counters/histograms.
  · routers for /v1/health, /v1/llm/*, /v1/embed.
  · ``GET /metrics`` (Prometheus) and ``GET /`` (service info).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from noetica import __version__
from noetica.serve.config import get_settings
from noetica.serve.logging import RequestContextMiddleware, configure_logging
from noetica.serve.metrics import MetricsMiddleware, render_latest
from noetica.serve.routes import embed, health, llm
from noetica.serve.security import ApiKeyMiddleware

settings = get_settings()
configure_logging(json_logs=settings.json_logs, level=settings.log_level)

app = FastAPI(
    title="noetica",
    version=__version__,
    description=(
        "Local FastAPI service over Ollama. Exposes a generic, domain-free "
        "structured-output primitive (/v1/llm/structured) plus /v1/embed and "
        "/v1/health. One of three front doors over a single local Ollama model "
        "registry; the others are chat (Open WebUI) and train + eval."
    ),
)

# add_middleware prepends, so the LAST added is the OUTERMOST. We want, outermost
# → innermost: request-context (so every log line and metric gets a request id
# and the access log wraps everything) → CORS (answers preflight, adds headers)
# → API key (cheap rejects) → metrics (closest to the route, sees the matched
# route template). Routers run innermost.
app.add_middleware(MetricsMiddleware)
app.add_middleware(ApiKeyMiddleware, api_key=settings.api_key)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=settings.cors_origins_list != ["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.add_middleware(RequestContextMiddleware)

app.include_router(health.router)
app.include_router(llm.router)
app.include_router(embed.router)


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return render_latest()


@app.get("/")
async def root() -> JSONResponse:
    s = get_settings()
    return JSONResponse(
        {
            "service": "noetica",
            "version": app.version,
            "ollama_url": s.ollama_url,
            "models": {
                "text": s.model_text,
                "vision": s.model_vision,
                "embed": s.model_embed,
            },
            "auth": "enabled" if s.api_key else "disabled",
            "cors_origins": s.cors_origins_list,
            "docs": "/docs",
            "health": "/v1/health",
            "metrics": "/metrics",
        }
    )
