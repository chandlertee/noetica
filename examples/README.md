# Examples

Runnable examples for each front door. Start the stack first: `./bin/up`
(or `noetica serve` for just the API). Override the API base with
`NOETICA_API=http://host:8001` if it isn't on localhost.

## Build on it — the API

| File | Endpoint | What it shows |
|------|----------|---------------|
| [`structured_extraction.py`](structured_extraction.py) | `POST /v1/llm/structured` | prompt + JSON Schema → validated JSON |
| [`embed_texts.py`](embed_texts.py) | `POST /v1/embed` | batch text → vectors + cosine similarity |
| [`rag_local.py`](rag_local.py) | both | a tiny, fully-local RAG loop (embed → retrieve → grounded structured answer) |

```sh
python examples/structured_extraction.py
python examples/embed_texts.py
python examples/rag_local.py "How do I expose the API safely?"
```

Health check (no body):

```sh
curl localhost:8001/v1/health
```

## Use it — chat

- [`chat_projects.md`](chat_projects.md) — zero → an Open WebUI **Workspace** that
  chats over your own documents (built-in RAG), with a custom model preset and a
  saved prompt. Fully local.

## Extend it — train → serve → chat → eval

- [`finetune_to_serve.md`](finetune_to_serve.md) — the full loop: QLoRA fine-tune →
  export to Ollama → the model appears in chat **and** the API → evaluate.
- [`data/sample_chat.jsonl`](data/sample_chat.jsonl) — a minimal chat dataset in the
  training format.
