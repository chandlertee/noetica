"""noetica.train — fine-tune a local model, then export it back to Ollama.

The *extend it* door. This subpackage carries the heavy, GPU-flavoured deps
(torch, transformers, trl/peft, unsloth) behind the ``[train]`` extra. The core
install (``noetica.serve``) never imports any of it.

Submodules:
  * :mod:`noetica.train.data`      — dataset prep + validation (stdlib only).
  * :mod:`noetica.train.config`    — training config (dataclass + YAML).
  * :mod:`noetica.train.finetune`  — the Unsloth QLoRA recipe.
  * :mod:`noetica.train.export_ollama` — merge → GGUF → quantize → Modelfile →
    ``ollama create``, so the tuned model is immediately servable + chattable.

The loop: chat → build on the API → fine-tune here → export to Ollama → the new
model appears in chat *and* the API → evaluate with :mod:`noetica.eval`.
"""
