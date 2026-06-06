#!/usr/bin/env python3
"""Call POST /v1/embed — batch text → embedding vectors (passthrough to Ollama).

python examples/embed_texts.py
"""

from __future__ import annotations

import os

import httpx

API = os.environ.get("NOETICA_API", "http://localhost:8001")


def main() -> None:
    resp = httpx.post(
        f"{API}/v1/embed",
        json={"texts": ["the quick brown fox", "a fast auburn vulpine"]},
        timeout=60,
    )
    resp.raise_for_status()
    body = resp.json()
    print(f"model: {body['model']}  dim: {body['dim']}  vectors: {len(body['embeddings'])}")
    # Cosine similarity of the two (near-synonymous) sentences.
    a, b = body["embeddings"]
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    print(f"cosine similarity: {dot / (na * nb):.3f}")


if __name__ == "__main__":
    main()
