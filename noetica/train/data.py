"""Dataset preparation + validation for fine-tuning.

Deliberately **stdlib only** (no torch) so you can validate a dataset on a
laptop before renting a GPU. The training format is a chat JSONL: one JSON
object per line with a ``messages`` array of ``{"role", "content"}`` turns.

    {"messages": [
      {"role": "system", "content": "You are a terse assistant."},
      {"role": "user", "content": "Capital of France?"},
      {"role": "assistant", "content": "Paris."}
    ]}

Validation checks structure, roles, ordering, and emptiness, and reports the
*line numbers* of bad records so they're easy to fix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

VALID_ROLES = {"system", "user", "assistant"}


@dataclass
class ValidationReport:
    total: int
    valid: int
    errors: list[str]  # "line N: <reason>"

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        head = f"{self.valid}/{self.total} records valid"
        if self.ok:
            return head + " ✓"
        shown = "\n".join(f"  {e}" for e in self.errors[:20])
        more = "" if len(self.errors) <= 20 else f"\n  … and {len(self.errors) - 20} more"
        return f"{head}\n{shown}{more}"


def _validate_record(obj: object) -> str | None:
    """Return an error string, or None if the record is valid."""
    if not isinstance(obj, dict) or "messages" not in obj:
        return "missing 'messages' array"
    messages = obj["messages"]
    if not isinstance(messages, list) or not messages:
        return "'messages' must be a non-empty array"

    roles = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            return f"message {i} is not an object"
        role, content = m.get("role"), m.get("content")
        if role not in VALID_ROLES:
            return f"message {i} has invalid role {role!r}"
        if not isinstance(content, str) or not content.strip():
            return f"message {i} has empty content"
        roles.append(role)

    # A trainable example must teach the model something — i.e. end on an
    # assistant turn with at least one user turn before it.
    if "assistant" not in roles:
        return "no assistant turn to learn from"
    if "user" not in roles:
        return "no user turn"
    if roles[-1] != "assistant":
        return "conversation must end on an assistant turn"
    return None


def validate_file(path: str | Path) -> ValidationReport:
    """Validate a chat JSONL file line by line."""
    path = Path(path)
    errors: list[str] = []
    total = 0
    valid = 0
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {lineno}: invalid JSON ({exc.msg})")
                continue
            err = _validate_record(obj)
            if err:
                errors.append(f"line {lineno}: {err}")
            else:
                valid += 1
    return ValidationReport(total=total, valid=valid, errors=errors)


def load_records(path: str | Path) -> list[dict]:
    """Load + validate, raising on the first structural problem.

    Returns the parsed records ready to hand to a chat-templating tokenizer.
    """
    report = validate_file(path)
    if not report.ok:
        raise ValueError(f"dataset {path} failed validation:\n{report.summary()}")
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def train_val_split(
    records: list[dict], val_fraction: float = 0.1
) -> tuple[list[dict], list[dict]]:
    """Deterministic tail split (no shuffling — keep it reproducible)."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    n_val = int(len(records) * val_fraction)
    if n_val == 0:
        return records, []
    return records[:-n_val], records[-n_val:]


def main(argv: list[str] | None = None) -> int:
    """`python -m noetica.train.data validate <file.jsonl>`."""
    import argparse

    parser = argparse.ArgumentParser(prog="noetica.train.data")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_val = sub.add_parser("validate", help="Validate a chat JSONL dataset.")
    p_val.add_argument("path")
    args = parser.parse_args(argv)

    report = validate_file(args.path)
    print(report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
