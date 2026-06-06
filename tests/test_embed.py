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
