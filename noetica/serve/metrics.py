"""Prometheus metrics.

Exposes a ``/metrics`` endpoint in the Prometheus text exposition format and a
middleware that records request counts and latencies. Labels use the matched
*route template* (e.g. ``/v1/llm/structured``) rather than the raw path, so
cardinality stays bounded even if a future route takes path parameters.
"""

from __future__ import annotations

import time

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# A dedicated registry keeps test runs isolated and avoids clobbering the
# global default registry if this module is imported more than once.
REGISTRY = CollectorRegistry()

REQUESTS = Counter(
    "noetica_requests_total",
    "Total HTTP requests handled.",
    labelnames=("method", "route", "status"),
    registry=REGISTRY,
)

LATENCY = Histogram(
    "noetica_request_duration_seconds",
    "HTTP request latency in seconds.",
    labelnames=("method", "route"),
    registry=REGISTRY,
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path or "unmatched"


class MetricsMiddleware(BaseHTTPMiddleware):
    """Count requests and observe latency, keyed by route template.

    The matched route is only present in the request scope *after* the inner
    app has run, so we resolve the label and observe both metrics once the
    response is in hand.
    """

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response: Response = await call_next(request)
        elapsed = time.perf_counter() - start
        route = _route_template(request)
        LATENCY.labels(request.method, route).observe(elapsed)
        REQUESTS.labels(request.method, route, str(response.status_code)).inc()
        return response


def render_latest() -> Response:
    """Return the current metrics in Prometheus exposition format."""
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
