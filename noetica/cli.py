"""``noetica`` command-line entry point.

Deliberately dependency-light: the ``serve`` and ``health`` subcommands import
only what the core needs. The ``train`` and ``eval`` subcommands import their
(heavier / optional) modules lazily, inside the handler, so ``noetica --help``
and ``noetica serve`` work on a laptop that never installed the ``[train]``
extra.
"""

from __future__ import annotations

import argparse
import sys

from noetica import __version__


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from noetica.serve.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "noetica.serve.main:app",
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=args.reload,
        log_config=None,  # our own structured logging is already configured
    )
    return 0


def _cmd_health(args: argparse.Namespace) -> int:
    import asyncio
    import json

    from noetica.serve.clients.ollama import OllamaClient, OllamaError
    from noetica.serve.config import get_settings

    settings = get_settings()
    expected = [settings.model_text, settings.model_vision, settings.model_embed]

    async def _check() -> dict:
        try:
            async with OllamaClient(settings.ollama_url, timeout=5.0) as client:
                models = await client.list_models()
        except OllamaError as exc:
            return {"ok": False, "ollama_reachable": False, "detail": str(exc)}
        bare = {name.split(":")[0] for name in models}
        have = set(models)
        present = {m: (m in have or m.split(":")[0] in bare) for m in expected}
        return {
            "ok": all(present.values()),
            "ollama_reachable": True,
            "models_present": present,
        }

    result = asyncio.run(_check())
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


def _cmd_eval(args: argparse.Namespace) -> int:
    from noetica.eval.run import main as eval_main

    return eval_main(args.extra)


def _cmd_export(args: argparse.Namespace) -> int:
    # Lives behind the [train] extra; import lazily so core installs don't need it.
    from noetica.train.export_ollama import main as export_main

    return export_main(args.extra)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="noetica", description="Local-AI stack over Ollama.")
    parser.add_argument("--version", action="version", version=f"noetica {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="Run the structured-output API (uvicorn).")
    p_serve.add_argument("--host", default=None, help="Bind host (default: NOETICA_HOST).")
    p_serve.add_argument(
        "--port", type=int, default=None, help="Bind port (default: NOETICA_PORT)."
    )
    p_serve.add_argument("--reload", action="store_true", help="Auto-reload on code changes.")
    p_serve.set_defaults(func=_cmd_serve)

    p_health = sub.add_parser("health", help="Check Ollama reachability + required models.")
    p_health.set_defaults(func=_cmd_health)

    p_eval = sub.add_parser("eval", help="Run the structured-output eval harness.")
    p_eval.add_argument(
        "extra", nargs=argparse.REMAINDER, help="Args forwarded to noetica.eval.run."
    )
    p_eval.set_defaults(func=_cmd_eval)

    p_export = sub.add_parser(
        "export", help="Export a fine-tuned adapter to Ollama (requires the [train] extra)."
    )
    p_export.add_argument(
        "extra", nargs=argparse.REMAINDER, help="Args forwarded to export_ollama."
    )
    p_export.set_defaults(func=_cmd_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
