from __future__ import annotations


def test_health_ok_when_models_present(client, mock_ollama):
    mock_ollama.get("/api/tags").respond(
        200,
        json={
            "models": [
                {"name": "qwen2.5:7b-instruct"},
                {"name": "qwen2.5vl:7b"},
                {"name": "nomic-embed-text"},
            ]
        },
    )
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["ollama_reachable"] is True
    assert all(body["models_present"].values())


def test_health_reports_missing_models(client, mock_ollama):
    mock_ollama.get("/api/tags").respond(200, json={"models": [{"name": "qwen2.5:7b-instruct"}]})
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["ollama_reachable"] is True
    assert body["models_present"]["qwen2.5:7b-instruct"] is True
    assert body["models_present"]["nomic-embed-text"] is False


def test_health_when_ollama_down(client, mock_ollama):
    from httpx import ConnectError

    mock_ollama.get("/api/tags").mock(side_effect=ConnectError("nope"))
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["ollama_reachable"] is False
