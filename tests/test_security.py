"""Tests for the optional API-key middleware."""

from __future__ import annotations

import os
from importlib import reload

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_with_key():
    """A fresh app instance with an API key configured.

    `get_settings` is lru_cached on the module, so we mutate the env then
    bust the cache + reload `main` to install the middleware with the key.
    """
    os.environ["NOETICA_API_KEY"] = "secret-key-123"
    # Reset settings cache + reload main so middleware picks up the new value.
    from noetica.serve import config, main

    config.get_settings.cache_clear()
    reload(main)
    with TestClient(main.app) as c:
        yield c
    # Tear down.
    del os.environ["NOETICA_API_KEY"]
    config.get_settings.cache_clear()
    reload(main)


def test_health_accessible_without_key(client_with_key, mock_ollama):
    mock_ollama.get("/api/tags").respond(200, json={"models": []})
    # No headers — health is in PUBLIC_PATHS.
    resp = client_with_key.get("/v1/health")
    assert resp.status_code == 200


def test_protected_endpoint_rejects_missing_key(client_with_key):
    resp = client_with_key.post(
        "/v1/llm/structured",
        json={
            "prompt": "x",
            "response_schema": {"type": "object"},
        },
    )
    assert resp.status_code == 401
    assert "missing" in resp.json()["detail"].lower()


def test_protected_endpoint_rejects_wrong_key(client_with_key):
    resp = client_with_key.post(
        "/v1/llm/structured",
        json={"prompt": "x", "response_schema": {"type": "object"}},
        headers={"X-API-Key": "nope"},
    )
    assert resp.status_code == 401


def test_protected_endpoint_accepts_correct_key(client_with_key, mock_ollama):
    from tests.conftest import ollama_generate_response

    mock_ollama.post("/api/generate").respond(200, json=ollama_generate_response({"x": 1}))
    resp = client_with_key.post(
        "/v1/llm/structured",
        json={
            "prompt": "x",
            "response_schema": {"type": "object", "properties": {"x": {"type": "integer"}}},
            "cache": False,
        },
        headers={"X-API-Key": "secret-key-123"},
    )
    assert resp.status_code == 200


def test_bearer_token_also_accepted(client_with_key, mock_ollama):
    from tests.conftest import ollama_generate_response

    mock_ollama.post("/api/generate").respond(200, json=ollama_generate_response({"x": 1}))
    resp = client_with_key.post(
        "/v1/llm/structured",
        json={
            "prompt": "x",
            "response_schema": {"type": "object", "properties": {"x": {"type": "integer"}}},
            "cache": False,
        },
        headers={"Authorization": "Bearer secret-key-123"},
    )
    assert resp.status_code == 200
