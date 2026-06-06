#!/usr/bin/env python3
"""A tiny, fully-local RAG loop over the Noetica API — no vector DB, no cloud.

Pipeline:
  1. /v1/embed   — embed a small in-memory corpus + the question.
  2. cosine top-k — retrieve the most relevant chunks (pure Python).
  3. /v1/llm/structured — answer grounded in the retrieved chunks, returning
     JSON with the answer *and* the source indices it used.

This is intentionally ~80 lines: it shows the two endpoints composing into
retrieval-augmented generation. For real corpora, point Open WebUI's built-in
RAG at a document collection (see examples/chat_projects.md) or swap step 2 for
a vector store.

    ./bin/up
    python examples/rag_local.py "How do I expose the API safely on my LAN?"
"""

from __future__ import annotations

import os
import sys

import httpx

API = os.environ.get("NOETICA_API", "http://localhost:8001")

CORPUS = [
    "Noetica exposes three endpoints under /v1: health, llm/structured, and embed.",
    "Set NOETICA_API_KEY to require an X-API-Key header before exposing the API on a LAN.",
    "Open WebUI is the chat front door; set WEBUI_AUTH=True before any network exposure.",
    "Ollama is the single local model registry shared by chat, the API, and evals.",
    "Fine-tune with noetica.train, export to Ollama, and the model appears in chat and the API.",
    "The disk cache is keyed by model + prompt version + payload; bump NOETICA_PROMPT_VERSION to bust it.",
]

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "sources": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Indices of the corpus chunks used.",
        },
    },
    "required": ["answer", "sources"],
    "additionalProperties": False,
}


def embed(texts: list[str]) -> list[list[float]]:
    resp = httpx.post(f"{API}/v1/embed", json={"texts": texts}, timeout=60)
    resp.raise_for_status()
    return resp.json()["embeddings"]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def main() -> None:
    question = " ".join(sys.argv[1:]) or "How do I protect the API before putting it on my network?"

    # 1 + 2: embed corpus + question, retrieve top-3 chunks by cosine similarity.
    vectors = embed(CORPUS + [question])
    *corpus_vecs, q_vec = vectors
    ranked = sorted(range(len(CORPUS)), key=lambda i: cosine(corpus_vecs[i], q_vec), reverse=True)
    top = ranked[:3]
    context = "\n".join(f"[{i}] {CORPUS[i]}" for i in top)
    print(f"Q: {question}\nretrieved chunks: {top}\n")

    # 3: grounded, structured answer.
    prompt = (
        "Answer the question using ONLY the numbered context. Cite the indices "
        f"you used in `sources`.\n\nContext:\n{context}\n\nQuestion: {question}"
    )
    resp = httpx.post(
        f"{API}/v1/llm/structured",
        json={"prompt": prompt, "response_schema": ANSWER_SCHEMA, "cache": False},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    print(f"A: {data['answer']}\nsources: {data['sources']}")


if __name__ == "__main__":
    main()
