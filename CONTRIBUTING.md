# Contributing to Noetica

Thanks for your interest. Noetica is a fully self-hosted, FOSS-first local-AI
stack — contributions that keep it local, light, and honest are very welcome.

## Principles

- **Local + FOSS only.** No proprietary cloud dependencies. Ollama, Open WebUI,
  Unsloth/TRL/PEFT, llama.cpp.
- **Keep the core light.** `noetica.serve` (and chat) must install and run on a
  laptop with **no** GPU/training deps. Heavy deps go behind an extra
  (`[train]`) and a compose profile — model your new module the same way (see
  [ROADMAP.md](ROADMAP.md) for how `agents/` will slot in).
- **No secrets, ever.** No keys, tokens, emails, PII, or hardcoded user paths in
  code, tests, or history. CI runs gitleaks.

## Dev setup

```sh
uv sync --extra dev          # core + test/lint/type tools (no torch)
uv run pytest                # tests (mocked Ollama; no live model)
uv run ruff check . && uv run ruff format --check .
uv run mypy
pre-commit install           # run the same checks on every commit
```

Optional, for the training module (needs an NVIDIA GPU):

```sh
uv sync --extra train
```

## Before you open a PR

- [ ] `ruff check` + `ruff format --check` clean
- [ ] `mypy` clean
- [ ] `pytest` green, with tests for new behaviour (mock Ollama via `respx` — no
      live model in the suite)
- [ ] eval gate passes: `python -m noetica.eval.run --check --ci`
- [ ] docs/examples updated if you changed the API or a workflow
- [ ] a `CHANGELOG.md` entry under `[Unreleased]`
- [ ] the core still installs without the `[train]` extra

## Commit + PR style

- Small, focused PRs with a clear description of *what* and *why*.
- Conventional-ish commit subjects are appreciated (`feat:`, `fix:`, `docs:`,
  `refactor:`) but not enforced.
- By contributing you agree your work is licensed under the project's
  [Apache-2.0](LICENSE) license.

## Reporting bugs / requesting features

Use the issue templates. For anything security-sensitive, follow
[SECURITY.md](SECURITY.md) instead of opening a public issue.
