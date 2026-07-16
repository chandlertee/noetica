"""Tests for the /v1/embed passthrough."""

from __future__ import annotations


def test_embed_returns_vectors_and_dim(client, mock_ollama):
    mock_ollama.post("/api/embed").respond(
        200, json={"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}
    )
    resp = client.post("/v1/embed", json={"texts": ["hello", "world"]})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["dim"] == 3
    assert len(body["embeddings"]) == 2
    assert body["model"]  # default embed model filled in


def test_embed_empty_input_short_circuits(client, mock_ollama):
    # No HTTP call should be needed for an empty batch.
    resp = client.post("/v1/embed", json={"texts": []})
    assert resp.status_code == 200
    body = resp.json()
    assert body["embeddings"] == []
    assert body["dim"] == 0


def test_embed_503_when_ollama_unreachable(client, mock_ollama):
    from httpx import ConnectError

    mock_ollama.post("/api/embed").mock(side_effect=ConnectError("down"))
    resp = client.post("/v1/embed", json={"texts": ["x"]})
    assert resp.status_code == 503


def test_embed_retries_cold_ollama_then_succeeds(client, mock_ollama):
    """First connection is refused; the retry succeeds → 200, not 503."""
    from httpx import ConnectError, Response

    calls = {"n": 0}

    def _flaky(req):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectError("connection refused")
        return Response(200, json={"embeddings": [[0.1, 0.2]]})

    mock_ollama.post("/api/embed").mock(side_effect=_flaky)
    resp = client.post("/v1/embed", json={"texts": ["x"]})
    assert resp.status_code == 200, resp.text
    assert calls["n"] == 2
    assert resp.json()["dim"] == 2
