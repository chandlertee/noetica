"""Tests for the generic /v1/llm/structured endpoint."""

from __future__ import annotations

from httpx import Response

from tests.conftest import ollama_generate_response

# Minimal "movie" schema — proves the endpoint isn't book-specific.
MOVIE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "year": {"type": "integer", "minimum": 1888},
        "director": {"type": "string"},
        "rating": {"type": "number", "minimum": 0.0, "maximum": 10.0},
    },
    "required": ["title", "year"],
    "additionalProperties": False,
}


def test_structured_validates_against_supplied_schema(client, mock_ollama):
    mock_ollama.post("/api/generate").respond(
        200,
        json=ollama_generate_response(
            {
                "title": "Dune",
                "year": 1984,
                "director": "David Lynch",
                "rating": 6.5,
            }
        ),
    )
    resp = client.post(
        "/v1/llm/structured",
        json={
            "prompt": "Extract movie info from: Dune (1984), directed by David Lynch.",
            "response_schema": MOVIE_SCHEMA,
            "cache": False,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["data"]["title"] == "Dune"
    assert body["data"]["year"] == 1984
    assert body["cached"] is False
    assert body["model"]  # filled with the default


def test_structured_repairs_after_invalid_output(client, mock_ollama):
    """First response violates the schema (year as string); second is valid."""
    responses = iter(
        [
            Response(
                200,
                json=ollama_generate_response(
                    {
                        "title": "Foo",
                        "year": "nineteen eighty four",
                    }
                ),
            ),
            Response(
                200,
                json=ollama_generate_response(
                    {
                        "title": "Foo",
                        "year": 1984,
                    }
                ),
            ),
        ]
    )
    mock_ollama.post("/api/generate").mock(side_effect=lambda req: next(responses))

    resp = client.post(
        "/v1/llm/structured",
        json={
            "prompt": "Extract movie info",
            "response_schema": MOVIE_SCHEMA,
            "cache": False,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["data"]["year"] == 1984


def test_structured_502_when_repair_fails(client, mock_ollama):
    """Both attempts violate the schema → 502."""
    responses = iter(
        [
            Response(200, json=ollama_generate_response({"title": "Foo", "year": "x"})),
            Response(200, json=ollama_generate_response({"title": "Foo", "year": "y"})),
        ]
    )
    mock_ollama.post("/api/generate").mock(side_effect=lambda req: next(responses))

    resp = client.post(
        "/v1/llm/structured",
        json={
            "prompt": "x",
            "response_schema": MOVIE_SCHEMA,
            "cache": False,
            "max_attempts": 2,
        },
    )
    assert resp.status_code == 502


def test_structured_503_when_ollama_unreachable(client, mock_ollama):
    from httpx import ConnectError

    mock_ollama.post("/api/generate").mock(side_effect=ConnectError("down"))
    resp = client.post(
        "/v1/llm/structured",
        json={
            "prompt": "x",
            "response_schema": MOVIE_SCHEMA,
            "cache": False,
        },
    )
    assert resp.status_code == 503


def test_structured_retries_cold_ollama_then_succeeds(client, mock_ollama):
    """First connection is refused (cold Ollama); the retry succeeds → 200, not 503."""
    from httpx import ConnectError

    calls = {"n": 0}

    def _flaky(req):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectError("connection refused")
        return Response(200, json=ollama_generate_response({"title": "Dune", "year": 1984}))

    mock_ollama.post("/api/generate").mock(side_effect=_flaky)
    resp = client.post(
        "/v1/llm/structured",
        json={"prompt": "x", "response_schema": MOVIE_SCHEMA, "cache": False},
    )
    assert resp.status_code == 200, resp.text
    assert calls["n"] == 2  # one failure + one successful retry
    assert resp.json()["data"]["title"] == "Dune"


def test_structured_overrides_default_model(client, mock_ollama):
    """Request can pin a specific model name."""
    captured: dict = {}

    def _capture(req):
        import json

        captured.update(json.loads(req.content))
        return Response(200, json=ollama_generate_response({"title": "x", "year": 2000}))

    mock_ollama.post("/api/generate").mock(side_effect=_capture)
    resp = client.post(
        "/v1/llm/structured",
        json={
            "model": "llama3.2:3b",
            "prompt": "x",
            "response_schema": MOVIE_SCHEMA,
            "cache": False,
        },
    )
    assert resp.status_code == 200
    assert captured["model"] == "llama3.2:3b"
    assert resp.json()["model"] == "llama3.2:3b"
