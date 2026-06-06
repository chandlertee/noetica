"""Structured JSON logging with per-request correlation IDs.

Every line emitted while handling a request carries the same ``request_id`` so
logs can be grepped/joined downstream. The id is taken from an inbound
``X-Request-ID`` header when present (so a reverse proxy or caller can thread
its own trace id through), otherwise a fresh uuid4 is minted. It is echoed back
on the response as ``X-Request-ID``.

Set ``NOETICA_JSON_LOGS=false`` to get human-readable logs in local dev.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Set for the duration of each request; read by the logging filter.
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

REQUEST_ID_HEADER = "X-Request-ID"


class _RequestIdFilter(logging.Filter):
    """Inject the current request id into every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        return True


class JsonFormatter(logging.Formatter):
    """Render a log record as a single JSON object on one line."""

    # Standard LogRecord attributes we don't want to duplicate into `extra`.
    _RESERVED = frozenset(logging.makeLogRecord({}).__dict__) | {
        "request_id",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Promote structured `extra=...` fields to top level.
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload.setdefault(key, value)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(*, json_logs: bool = True, level: str = "INFO") -> None:
    """Install a single stdout handler with the chosen formatter.

    Idempotent: replaces any handlers we previously installed so re-import or
    ``uvicorn --reload`` doesn't stack duplicate handlers.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_RequestIdFilter())
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s")
        )

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    # Let uvicorn's loggers propagate to root instead of double-printing.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, log access lines, and time each request."""

    def __init__(self, app, logger_name: str = "noetica.access") -> None:
        super().__init__(app)
        self._log = logging.getLogger(logger_name)

    async def dispatch(self, request: Request, call_next):
        incoming = request.headers.get(REQUEST_ID_HEADER)
        rid = incoming or uuid.uuid4().hex
        token = request_id_ctx.set(rid)
        start = time.perf_counter()
        # Log *before* the finally resets the context var, so every line —
        # including this access line — carries the request id.
        try:
            response: Response = await call_next(request)
            elapsed_ms = (time.perf_counter() - start) * 1000
            response.headers[REQUEST_ID_HEADER] = rid
            self._log.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round(elapsed_ms, 2),
                },
            )
            return response
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._log.exception(
                "request failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(elapsed_ms, 2),
                },
            )
            raise
        finally:
            request_id_ctx.reset(token)
