## What & why

What does this change, and why?

## Door(s) touched

- [ ] chat (Open WebUI)
- [ ] serve (API)
- [ ] train
- [ ] eval
- [ ] docs / infra

## Checklist

- [ ] `uv run ruff check .` and `uv run ruff format --check .` pass
- [ ] `uv run mypy` passes
- [ ] `uv run pytest` passes, with tests for new behaviour (Ollama mocked via respx)
- [ ] `python -m noetica.eval.run --check --ci` passes
- [ ] Core still installs/runs **without** the `[train]` extra (no torch/CUDA in core)
- [ ] No secrets, PII, or hardcoded user paths
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] Docs/examples updated if the API or a workflow changed

## Notes for reviewers

Anything worth calling out (trade-offs, follow-ups).
