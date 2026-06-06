"""Disk cache for prompt responses.

Keyed by sha256 of (model name + prompt version + canonical-JSON payload).
Cache hits short-circuit the Ollama call, which is the slow path. Cache is
versioned by `prompt_version` from settings — bumping it invalidates all
entries derived from changed prompts.

Atomicity: writes go to a temp file in the same directory, then atomically
renamed into place. Reads tolerate missing/partial files (they return None).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _key(model: str, prompt_version: str, payload: Any) -> str:
    canonical = json.dumps(
        {"m": model, "v": prompt_version, "p": payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _path_for(cache_dir: Path, key: str) -> Path:
    # Shard by first 2 chars to avoid one big directory.
    return cache_dir / key[:2] / f"{key}.json"


def get(cache_dir: Path, model: str, prompt_version: str, payload: Any) -> dict | None:
    if not cache_dir:
        return None
    key = _key(model, prompt_version, payload)
    path = _path_for(cache_dir, key)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def put(cache_dir: Path, model: str, prompt_version: str, payload: Any, value: dict) -> None:
    if not cache_dir:
        return
    key = _key(model, prompt_version, payload)
    path = _path_for(cache_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
