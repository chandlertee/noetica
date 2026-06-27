"""Shared test fixtures.

Tests run against the FastAPI app using `httpx.ASGITransport` (no network).
Ollama is mocked at the HTTP layer via `respx` so we never hit a real model.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
import respx
from fastapi.testclient import TestClient


# Force a clean cache dir BEFORE the app imports settings.
@pytest.fixture(scope="session", autouse=True)
def _isolated_cache():
    tmp = Path(tempfile.mkdtemp(prefix="noetica-tests-"))
    os.environ["NOETICA_CACHE_DIR"] = str(tmp)
    os.environ["OLLAMA_URL"] = "http://test-ollama:11434"
    os.environ["NOETICA_CACHE_ENABLED"] = "false"
    # Exercise the retry path without real backoff sleeps slowing the suite.
    os.environ["OLLAMA_RETRY_BASE_DELAY"] = "0"
    # Keep test output readable rather than emitting JSON access logs.
    os.environ["NOETICA_JSON_LOGS"] = "false"
    yield tmp


@pytest.fixture
def client():
    # Import inside the fixture so the env vars above take effect first.
    from noetica.serve.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_ollama():
    """Mock the Ollama HTTP API; yields a respx router for configuration."""
    with respx.mock(base_url="http://test-ollama:11434", assert_all_called=False) as router:
        yield router


def ollama_generate_response(payload: dict) -> dict:
    """Build a `/api/generate` response whose `response` field is `payload` as JSON."""
    return {"response": json.dumps(payload), "done": True}
