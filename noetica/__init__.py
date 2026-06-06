"""Noetica — a fully self-hosted local-AI stack over a single Ollama registry.

Three front doors share one model registry:

  * **chat**  — Open WebUI (use it)
  * **serve** — :mod:`noetica.serve`, the structured-output API (build on it)
  * **train + eval** — :mod:`noetica.train` / :mod:`noetica.eval` (extend it)

The top-level package and ``noetica.serve`` stay light: importing them never
pulls torch/CUDA. Training deps live behind the ``[train]`` extra.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
