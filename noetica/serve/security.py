"""Optional API-key authentication.

Off by default (empty `NOETICA_API_KEY` → middleware is a no-op). When set,
every `/v1/*` request must carry the matching key in one of:

  · `X-API-Key: <key>` header
  · `Authorization: Bearer <key>` header

The check is short-circuited for `/v1/health` so monitors don't need the key,
and for unauthenticated paths (`/`, `/docs`, `/openapi.json`).

We do constant-time comparison on the key to avoid leaking via timing.
"""

from __future__ import annotations

import hmac
import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


# Paths that are reachable without an API key even when one is configured.
PUBLIC_PATHS: frozenset[str] = frozenset(
    {
        "/",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/v1/health",
    }
)


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Reject /v1/* requests that don't carry the configured API key.

    Constructed with the configured key. An empty key disables enforcement —
    we install the middleware anyway so the same code path runs in dev and
    prod, just with a no-op.
    """

    def __init__(self, app: ASGIApp, api_key: str = "") -> None:
        super().__init__(app)
        self._key = api_key

    async def dispatch(self, request: Request, call_next):
        # No key configured → middleware is a no-op.
        if not self._key:
            return await call_next(request)
        # Public paths (health, docs, root) skip the check.
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)
        # Static asset / favicon-style requests also bypass.
        if not request.url.path.startswith("/v1/"):
            return await call_next(request)

        supplied = self._extract_key(request)
        if not supplied:
            return _unauthorized("missing API key (set X-API-Key header)")
        if not hmac.compare_digest(supplied, self._key):
            return _unauthorized("invalid API key")

        return await call_next(request)

    @staticmethod
    def _extract_key(request: Request) -> str:
        # Prefer X-API-Key (explicit), fall back to Bearer token for OAuth-style clients.
        api_key = request.headers.get("x-api-key")
        if api_key:
            return api_key
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return ""


def _unauthorized(detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={"error": "unauthorized", "detail": detail, "kind": "validation"},
        headers={"WWW-Authenticate": 'Bearer realm="noetica"'},
    )
