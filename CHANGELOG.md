# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] — 2026-06-06

First public release. Noetica is a fully self-hosted local-AI stack over a single
Ollama model registry, with three front doors: **chat** (use it), a
**structured-output API** (build on it), and **train + eval** (extend it).

### Added

- **Chat (Open WebUI)** as a first-class front door. `bin/up` brings up a working
  private chat workbench out of the box on laptop (`cpu`) and GPU (`full`)
  profiles. Walkthrough for Workspaces, document collections / RAG, custom model
  presets, and the prompt library in [`examples/chat_projects.md`](examples/chat_projects.md).
- **Serve API** (`noetica.serve`): `POST /v1/llm/structured` (prompt + JSON Schema
  → validated JSON, with a repair-retry loop and disk cache), `POST /v1/embed`,
  and `GET /v1/health`.
  - Structured JSON logging with per-request `X-Request-ID` correlation.
  - Prometheus metrics at `GET /metrics` (request counts + latency histograms,
    labeled by route template).
  - Optional API-key auth (`NOETICA_API_KEY`) and configurable CORS.
  - Full type coverage; deterministic tests with `respx` (no live Ollama).
- **Train** (`noetica.train`, optional `[train]` extra + `train` compose profile):
  a runnable Unsloth QLoRA recipe, stdlib-only dataset prep/validation, and
  `export_ollama.py` (merge adapter → GGUF via llama.cpp → quantize → Modelfile →
  `ollama create`) so a fine-tuned model is immediately servable and chattable.
  Colab notebook for the no-GPU path.
- **Eval** (`noetica.eval`): golden cases for the structured endpoint, a
  `case × model` comparison harness (`run.py`), and a deterministic CI regression
  gate (`--check --ci`).
- **CLI**: `noetica serve | health | eval | export` (and `python -m noetica`).
- OSS hygiene: Apache-2.0 license, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY,
  AGENTS.md, ROADMAP, issue/PR templates, Dependabot, pre-commit, and CI
  (ruff lint+format, mypy, pytest+coverage, docker build, gitleaks; Python
  3.11 + 3.12).

[Unreleased]: https://github.com/chandlertee/noetica/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/chandlertee/noetica/releases/tag/v1.0.0
