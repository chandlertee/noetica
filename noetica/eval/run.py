"""Golden-case eval harness for /v1/llm/structured.

A *case* is a JSON file under ``cases/`` describing a prompt, a response schema,
and what a correct answer looks like. The harness scores model output against
the case and prints a comparison table across model variants.

Usage
-----
Validate the corpus + scorer with no live model (the CI regression gate)::

    python -m noetica.eval.run --check

Score the golden fixtures end-to-end, still offline::

    python -m noetica.eval.run --offline

Run against a live API for one or more models and compare::

    python -m noetica.eval.run --api http://localhost:8001 --model qwen2.5:7b-instruct --model llama3.1:latest

Exit code is non-zero if any selected case fails — so ``--check`` doubles as a
pre-merge gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CASES_DIR = Path(__file__).parent / "cases"

# What counts as "correct" for a case.
#   schema_only — output just has to validate against the schema
#   fields      — listed fields must deep-equal the expected values
#   exact       — the whole object must deep-equal `expected`
MatchMode = str


@dataclass(frozen=True)
class Case:
    name: str
    prompt: str
    response_schema: dict[str, Any]
    match: MatchMode = "schema_only"
    expected: Any = None
    match_fields: list[str] = field(default_factory=list)
    ci: bool = False
    fixture: Any = None  # recorded model output, used in offline mode
    description: str = ""

    @classmethod
    def from_file(cls, path: Path) -> Case:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            name=raw.get("name", path.stem),
            prompt=raw["prompt"],
            response_schema=raw["response_schema"],
            match=raw.get("match", "schema_only"),
            expected=raw.get("expected"),
            match_fields=raw.get("match_fields", []),
            ci=raw.get("ci", False),
            fixture=raw.get("fixture"),
            description=raw.get("description", ""),
        )


@dataclass
class Result:
    case: str
    model: str
    ok: bool
    detail: str = ""


def load_cases(directory: Path = CASES_DIR) -> list[Case]:
    cases = [Case.from_file(p) for p in sorted(directory.glob("*.json"))]
    if not cases:
        raise SystemExit(f"no eval cases found in {directory}")
    return cases


def schema_errors(schema: dict[str, Any], data: Any) -> list[str]:
    """Return human-readable JSON-Schema validation errors (empty == valid)."""
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(schema)
    return [
        f"{'.'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
        for e in validator.iter_errors(data)
    ]


def score(case: Case, data: Any) -> Result:
    """Score model `data` against a case. Pure; no I/O."""
    errs = schema_errors(case.response_schema, data)
    if errs:
        return Result(case.name, "-", ok=False, detail=f"schema: {errs[0]}")

    if case.match == "schema_only":
        return Result(case.name, "-", ok=True, detail="schema ok")

    if case.match == "exact":
        ok = data == case.expected
        return Result(case.name, "-", ok=ok, detail="exact match" if ok else "≠ expected")

    if case.match == "fields":
        if not isinstance(data, dict):
            return Result(case.name, "-", ok=False, detail="expected an object")
        mismatched = [f for f in case.match_fields if data.get(f) != (case.expected or {}).get(f)]
        ok = not mismatched
        return Result(case.name, "-", ok=ok, detail="fields match" if ok else f"≠ {mismatched}")

    return Result(case.name, "-", ok=False, detail=f"unknown match mode: {case.match}")


def call_api(api: str, model: str, case: Case, timeout: float) -> Any:
    """Call a live /v1/llm/structured endpoint and return the parsed `data`."""
    import httpx

    resp = httpx.post(
        f"{api.rstrip('/')}/v1/llm/structured",
        json={
            "model": model,
            "prompt": case.prompt,
            "response_schema": case.response_schema,
            "cache": False,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["data"]


def run_offline(cases: list[Case], *, ci_only: bool) -> list[Result]:
    selected = [c for c in cases if c.ci or not ci_only]
    results: list[Result] = []
    for case in selected:
        if case.fixture is None:
            results.append(Result(case.name, "fixture", ok=False, detail="no fixture recorded"))
            continue
        r = score(case, case.fixture)
        r.model = "fixture"
        results.append(r)
    return results


def run_live(cases: list[Case], models: list[str], api: str, timeout: float) -> list[Result]:
    results: list[Result] = []
    for model in models:
        for case in cases:
            try:
                data = call_api(api, model, case, timeout)
            except Exception as exc:  # network / upstream / schema 502
                results.append(Result(case.name, model, ok=False, detail=f"error: {exc}"))
                continue
            r = score(case, data)
            r.model = model
            results.append(r)
    return results


def check_corpus(cases: list[Case]) -> list[Result]:
    """Validate the corpus itself: schemas well-formed, fixtures self-consistent.

    This is the CI gate's real value — it guards the golden contract without a
    model: every schema must be a valid Draft 2020-12 schema, and every recorded
    fixture/expected must satisfy its own case.
    """
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError

    results: list[Result] = []
    for case in cases:
        try:
            Draft202012Validator.check_schema(case.response_schema)
        except SchemaError as exc:
            results.append(
                Result(case.name, "schema", ok=False, detail=f"invalid schema: {exc.message}")
            )
            continue
        if case.fixture is not None:
            r = score(case, case.fixture)
            r.model = "fixture"
            results.append(r)
        else:
            results.append(Result(case.name, "schema", ok=True, detail="schema valid (no fixture)"))
    return results


def print_table(results: list[Result]) -> None:
    models = sorted({r.model for r in results})
    cases = sorted({r.case for r in results})
    by_key = {(r.case, r.model): r for r in results}

    name_w = max([len("case"), *(len(c) for c in cases)])
    col_w = max(10, *(len(m) for m in models))

    header = "case".ljust(name_w) + "  " + "  ".join(m.ljust(col_w) for m in models)
    print(header)
    print("-" * len(header))
    for case in cases:
        row = [case.ljust(name_w)]
        for model in models:
            r = by_key.get((case, model))
            cell = "—" if r is None else ("PASS" if r.ok else "FAIL")
            row.append(cell.ljust(col_w))
        print("  ".join(row))
    print("-" * len(header))
    passed = sum(1 for r in results if r.ok)
    print(f"{passed}/{len(results)} checks passed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="noetica.eval.run", description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Validate the corpus + scorer (no model). CI gate."
    )
    parser.add_argument(
        "--offline", action="store_true", help="Score recorded fixtures (no model)."
    )
    parser.add_argument("--ci", action="store_true", help="Restrict to ci-tagged cases.")
    parser.add_argument("--api", default="http://localhost:8001", help="Base URL of a live API.")
    parser.add_argument(
        "--model", action="append", dest="models", help="Model to evaluate (repeatable)."
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="Per-call timeout (live mode)."
    )
    args = parser.parse_args(argv)

    cases = load_cases()

    if args.check:
        results = check_corpus([c for c in cases if c.ci or not args.ci])
    elif args.offline:
        results = run_offline(cases, ci_only=args.ci)
    else:
        models = args.models or ["qwen2.5:7b-instruct"]
        results = run_live(cases, models, args.api, args.timeout)

    print_table(results)
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
