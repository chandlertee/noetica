# AGENTS.md

Guidance for coding agents (and humans) working in this repo. Keep changes
consistent with the structure and constraints below.

## What this is

Noetica: a self-hosted local-AI stack over one Ollama registry, with three front
doors — **chat** (Open WebUI), **serve** (a structured-output FastAPI), and
**train + eval**. The unifying loop: chat → build on the API → fine-tune → export
to Ollama → it appears in chat + the API → evaluate.

## Layout

```
noetica/
  serve/      FastAPI app — the /v1 API. LIGHT: no torch/CUDA. Core deps only.
    main.py, config.py, models.py, deps.py, cache.py, security.py,
    logging.py (JSON logs + request IDs), metrics.py (Prometheus),
    clients/ollama.py, routes/{health,llm,embed}.py
  train/      QLoRA fine-tune + export to Ollama. HEAVY: behind the [train] extra.
              data.py + config.py are stdlib-only (importable without torch).
  eval/       Golden-case harness for the structured endpoint. Core deps only.
  cli.py      `noetica serve|health|eval|export`. Lazy-imports heavy paths.
examples/     Runnable per-endpoint examples, local RAG, chat + finetune docs.
tests/        pytest; Ollama mocked via respx (no live model).
```

## Hard constraints

1. **Keep `noetica.serve` (+ chat) installable and runnable without a GPU.** Never
   add torch/CUDA/training deps to core `dependencies`. Heavy deps go under the
   `[train]` extra and a compose profile. New modules follow the same pattern
   (see `agents/` in [ROADMAP.md](ROADMAP.md)).
2. **No secrets, PII, or hardcoded user paths** anywhere in the tree or history.
3. **FOSS-first**, local-only. No proprietary cloud dependencies.
4. **Heavy imports stay lazy** — import torch/unsloth/peft *inside* functions so
   modules import (and `--help`/tests work) without the `[train]` extra.

## Commands

```sh
uv sync --extra dev          # set up (core + dev tools, no torch)
uv run pytest                # tests + coverage
uv run ruff check . && uv run ruff format --check .
uv run mypy                  # type-checks serve + eval + cli (train excluded)
python -m noetica.eval.run --check --ci   # eval regression gate (no model)
./bin/up                     # bring up chat + API (+ Ollama on GPU hosts)
```

## Conventions

- Python ≥ 3.11, `from __future__ import annotations`, full type hints on serve.
- Config via env, prefix `NOETICA_` (service) / unprefixed `OLLAMA_URL`,
  `MODEL_*` (runtime). pydantic-settings in `serve/config.py`.
- Tests are deterministic: mock Ollama at the HTTP layer with `respx`; never call
  a real model in the suite.
- The public API contract is `/v1/llm/structured`, `/v1/embed`, `/v1/health`
  (+ `/metrics`). Keep it domain-free — domain logic belongs in callers.
- Update `CHANGELOG.md` (`[Unreleased]`) and any affected example/doc with code
  changes.
