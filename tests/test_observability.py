"""Tests for the observability surface: request IDs, access logs, /metrics."""

from __future__ import annotations

from noetica.serve.logging import REQUEST_ID_HEADER


def test_request_id_minted_when_absent(client, mock_ollama):
    mock_ollama.get("/api/tags").respond(200, json={"models": []})
    resp = client.get("/v1/health")
    rid = resp.headers.get(REQUEST_ID_HEADER)
    assert rid and len(rid) >= 16  # uuid4 hex


def test_inbound_request_id_is_propagated(client, mock_ollama):
    mock_ollama.get("/api/tags").respond(200, json={"models": []})
    resp = client.get("/v1/health", headers={REQUEST_ID_HEADER: "trace-abc-123"})
    assert resp.headers.get(REQUEST_ID_HEADER) == "trace-abc-123"


def test_access_log_is_emitted_within_request_context(client, mock_ollama):
    """Regression: the access line must be logged before the context var resets,
    so it carries the request id (not the '-' placeholder)."""
    import logging

    from noetica.serve.logging import request_id_ctx

    seen: list[tuple[str, str]] = []

    class _Cap(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            # Capture the context var value at the moment the line is logged.
            seen.append((record.getMessage(), request_id_ctx.get()))

    access = logging.getLogger("noetica.access")
    handler = _Cap()
    access.addHandler(handler)
    try:
        mock_ollama.get("/api/tags").respond(200, json={"models": []})
        client.get("/v1/health", headers={REQUEST_ID_HEADER: "rid-xyz"})
    finally:
        access.removeHandler(handler)

    assert ("request", "rid-xyz") in seen


def test_metrics_endpoint_exposes_prometheus_text(client, mock_ollama):
    mock_ollama.get("/api/tags").respond(200, json={"models": []})
    # Generate at least one recorded request first.
    client.get("/v1/health")

    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    body = resp.text
    assert "noetica_requests_total" in body
    assert "noetica_request_duration_seconds" in body
    # The matched route template should appear as a label, not the raw path.
    assert 'route="/v1/health"' in body


def test_metrics_label_uses_route_template(client, mock_ollama):
    mock_ollama.get("/api/tags").respond(200, json={"models": []})
    client.get("/v1/health")
    body = client.get("/metrics").text
    # Counter line for the health route with a 200 status.
    assert 'noetica_requests_total{method="GET",route="/v1/health",status="200"}' in body
