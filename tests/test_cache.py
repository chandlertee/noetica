"""Tests for the prompt-version-keyed disk cache."""

from __future__ import annotations

from pathlib import Path

from noetica.serve import cache as cache_mod


def test_put_then_get_roundtrips(tmp_path: Path):
    payload = {"prompt": "hello", "schema": {"type": "object"}}
    cache_mod.put(tmp_path, "qwen2.5", "v1", payload, {"answer": 42})
    got = cache_mod.get(tmp_path, "qwen2.5", "v1", payload)
    assert got == {"answer": 42}


def test_miss_returns_none(tmp_path: Path):
    assert cache_mod.get(tmp_path, "qwen2.5", "v1", {"p": "nope"}) is None


def test_prompt_version_namespaces_entries(tmp_path: Path):
    payload = {"prompt": "hello"}
    cache_mod.put(tmp_path, "qwen2.5", "v1", payload, {"x": 1})
    # Same payload + model but a bumped prompt version → different key → miss.
    assert cache_mod.get(tmp_path, "qwen2.5", "v2", payload) is None


def test_model_namespaces_entries(tmp_path: Path):
    payload = {"prompt": "hello"}
    cache_mod.put(tmp_path, "model-a", "v1", payload, {"x": 1})
    assert cache_mod.get(tmp_path, "model-b", "v1", payload) is None


def test_corrupt_file_is_treated_as_miss(tmp_path: Path):
    payload = {"prompt": "hello"}
    cache_mod.put(tmp_path, "qwen2.5", "v1", payload, {"x": 1})
    # Corrupt every cached json file.
    for p in tmp_path.rglob("*.json"):
        p.write_text("{ not valid json")
    assert cache_mod.get(tmp_path, "qwen2.5", "v1", payload) is None
