"""Tests for the JSON log formatter and request-id context."""

from __future__ import annotations

import json
import logging

from noetica.serve.logging import JsonFormatter, request_id_ctx


def _record(msg: str, **extra) -> logging.LogRecord:
    rec = logging.LogRecord(
        name="noetica.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )
    for k, v in extra.items():
        setattr(rec, k, v)
    return rec


def test_formatter_emits_single_json_object():
    rec = _record("hello")
    rec.request_id = "abc123"
    line = JsonFormatter().format(rec)
    obj = json.loads(line)  # one line, valid JSON
    assert obj["msg"] == "hello"
    assert obj["level"] == "INFO"
    assert obj["logger"] == "noetica.test"
    assert obj["request_id"] == "abc123"
    assert obj["ts"].endswith("Z")


def test_formatter_promotes_extra_fields():
    rec = _record("request", request_id="r1", status=200, path="/v1/health")
    obj = json.loads(JsonFormatter().format(rec))
    assert obj["status"] == 200
    assert obj["path"] == "/v1/health"


def test_request_id_ctx_defaults_to_dash():
    # Outside a request, the context var carries a placeholder.
    assert request_id_ctx.get() == "-"
