# Chat over your own documents — zero → working project (local)

Open WebUI is a first-class Noetica front door, not just a container. This walks
from nothing to a private **Workspace** that chats over your files, with a custom
model preset and a saved prompt — all on your machine, no cloud.

Prereqts: `./bin/up` is running and the chat UI is at <http://localhost:3000>.

## 0 · First run

1. Open <http://localhost:3000>. With `WEBUI_AUTH=False` (the default) you go
   straight in; otherwise create the first admin account (it stays local).
2. Top-left model picker → pick a model already pulled in Ollama
   (`qwen2.5:7b-instruct`). If the list is empty, pull one:
   ```sh
   ollama pull qwen2.5:7b-instruct
   ```
3. Say hello. You now have a private ChatGPT-style workbench against a local model.

## 1 · Make a Workspace + Knowledge collection (RAG over your files)

"Chat with your files" in Open WebUI is built-in RAG: documents are chunked,
embedded, and retrieved at query time. Inference and embeddings both run through
your local Ollama — nothing leaves the box.

1. Left sidebar → **Workspace** → **Knowledge** → **+ Create a knowledge base**.
   Name it e.g. `my-docs`.
2. **Add content** → upload PDFs / Markdown / text, or point it at a folder.
   Wait for indexing (status shows per file).
3. New chat → in the message box type `#` and select `my-docs` (or `#` then the
   filename for a single doc). Ask a question about your documents.
4. Answers now cite retrieved snippets. Tune retrieval under
   **Admin Settings → Documents** (embedding model, chunk size, top-k). Set the
   embedding model to `nomic-embed-text` to match the API's default.

> Tip: the same `nomic-embed-text` model backs both this RAG and the API's
> `/v1/embed`, so retrieval behaves consistently across the chat and build doors.

## 2 · A custom model preset (system prompt + params)

Presets let you save a model + system prompt + parameters as a reusable
"assistant".

1. **Workspace → Models → + Create a model**.
2. Base model: `qwen2.5:7b-instruct`. Name: `docs-helper`.
3. System prompt:
   ```
   You answer strictly from the attached knowledge. If the answer isn't in the
   documents, say so. Be concise and cite the source file.
   ```
4. Optionally attach the `my-docs` knowledge base so this assistant always has it.
5. Set parameters (e.g. temperature `0.2`) and save. `docs-helper` now appears in
   the model picker.

## 3 · Save a prompt in the prompt library

1. **Workspace → Prompts → + Create a prompt**.
2. Give it a slash trigger, e.g. `/summarize`, with content:
   ```
   Summarize the attached documents as 5 bullet points, each with the source file.
   ```
3. In any chat, type `/summarize` to expand it. Build up a library of your
   common asks.

## 4 · Bring in a model you fine-tuned

After [`finetune_to_serve.md`](finetune_to_serve.md), your `ollama create`d model
shows up in the **same** model picker. Point a preset's base model at it to chat
with your own fine-tune over your own docs — the full loop, closed, locally.

## Security note

Defaults are tuned for a single-user localhost box. Before exposing chat on a LAN:
set `WEBUI_AUTH=True`, put it behind a TLS reverse proxy, and set `NOETICA_API_KEY`
for the API. See [SECURITY.md](../SECURITY.md).
