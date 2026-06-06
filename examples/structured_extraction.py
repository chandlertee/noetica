#!/usr/bin/env python3
"""Call POST /v1/llm/structured — prompt + JSON Schema → validated JSON.

The whole point of the `serve` door: you describe the shape you want with a JSON
Schema and get back JSON that conforms (the service retries/repairs once on
drift, validates server-side, and caches).

    ./bin/up                       # or: noetica serve
    python examples/structured_extraction.py
"""

from __future__ import annotations

import json
import os

import httpx

API = os.environ.get("NOETICA_API", "http://localhost:8001")

MOVIE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "year": {"type": "integer", "minimum": 1888},
        "director": {"type": "string"},
        "genres": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "year"],
    "additionalProperties": False,
}


def main() -> None:
    resp = httpx.post(
        f"{API}/v1/llm/structured",
        json={
            # "model": "qwen2.5:7b-instruct",   # optional; defaults to MODEL_TEXT
            "prompt": "Describe the film Arrival (2016) by Denis Villeneuve.",
            "response_schema": MOVIE_SCHEMA,
        },
        timeout=120,
    )
    resp.raise_for_status()
    body = resp.json()
    print(f"model:  {body['model']}  (cached={body['cached']})")
    print(json.dumps(body["data"], indent=2))


if __name__ == "__main__":
    main()
