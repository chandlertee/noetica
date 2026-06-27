"""Unit tests for the OllamaClient module-level helpers.

`with_retry` is pure control-flow, so we test it directly rather than only
through the routes. The behavior that matters most here is the deliberate
asymmetry: connection failures are retried, timeouts are not.
"""

from __future__ import annotations

import pytest

from noetica.serve.clients import ollama as ollama_mod
from noetica.serve.clients.ollama import (
    OllamaSchemaError,
    OllamaTimeout,
    OllamaUnavailable,
    with_retry,
)


def _counter(results):
    """Build a zero-arg async fn that yields each item in `results` per call.

    An item that is an exception instance is raised; anything else is returned.
    Also records how many times it was invoked.
    """
    calls = {"n": 0}
    it = iter(results)

    async def fn():
        calls["n"] += 1
        item = next(it)
        if isinstance(item, BaseException):
            raise item
        return item

    return fn, calls


async def test_returns_value_without_retrying_on_success():
    fn, calls = _counter(["ok"])
    assert await with_retry(fn, retries=2, base_delay=0) == "ok"
    assert calls["n"] == 1


async def test_retries_unavailable_then_succeeds():
    fn, calls = _counter([OllamaUnavailable("cold"), OllamaUnavailable("cold"), "ok"])
    assert await with_retry(fn, retries=2, base_delay=0) == "ok"
    assert calls["n"] == 3  # two failures + one success


async def test_raises_last_unavailable_after_exhausting_retries():
    final = OllamaUnavailable("still down")
    fn, calls = _counter([OllamaUnavailable("down"), final])
    with pytest.raises(OllamaUnavailable) as exc:
        await with_retry(fn, retries=1, base_delay=0)
    assert exc.value is final
    assert calls["n"] == 2  # initial attempt + one retry, both fail


async def test_timeout_is_not_retried():
    """A slow generation should surface immediately as a timeout, never re-run."""
    fn, calls = _counter([OllamaTimeout("too slow"), "unreached"])
    with pytest.raises(OllamaTimeout):
        await with_retry(fn, retries=3, base_delay=0)
    assert calls["n"] == 1


async def test_non_ollama_unavailable_errors_propagate_immediately():
    fn, calls = _counter([OllamaSchemaError("bad schema"), "unreached"])
    with pytest.raises(OllamaSchemaError):
        await with_retry(fn, retries=3, base_delay=0)
    assert calls["n"] == 1


async def test_retries_zero_disables_retry():
    fn, calls = _counter([OllamaUnavailable("down"), "unreached"])
    with pytest.raises(OllamaUnavailable):
        await with_retry(fn, retries=0, base_delay=0)
    assert calls["n"] == 1


async def test_backoff_doubles_each_attempt(monkeypatch):
    delays: list[float] = []

    async def fake_sleep(d):
        delays.append(d)

    monkeypatch.setattr(ollama_mod.asyncio, "sleep", fake_sleep)

    fn, _ = _counter([OllamaUnavailable("1"), OllamaUnavailable("2"), "ok"])
    assert await with_retry(fn, retries=3, base_delay=0.5) == "ok"
    # base_delay * 2**attempt for attempts 0 and 1; no sleep after the success.
    assert delays == [0.5, 1.0]
