"""Tests for the service-info root and the CLI parser."""

from __future__ import annotations

from noetica import __version__
from noetica.cli import build_parser


def test_root_reports_service_info(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "noetica"
    assert body["version"] == __version__
    assert body["health"] == "/v1/health"
    assert body["metrics"] == "/metrics"
    assert set(body["models"]) == {"text", "vision", "embed"}


def test_cli_parses_serve_defaults():
    parser = build_parser()
    args = parser.parse_args(["serve"])
    assert args.command == "serve"
    assert args.reload is False


def test_cli_parses_health():
    parser = build_parser()
    args = parser.parse_args(["health"])
    assert args.command == "health"
    assert callable(args.func)
