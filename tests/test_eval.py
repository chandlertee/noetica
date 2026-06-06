"""Tests for the eval harness scorer + corpus integrity."""

from __future__ import annotations

from noetica.eval.run import Case, check_corpus, load_cases, score


def _movie_case(**over) -> Case:
    base = dict(
        name="t",
        prompt="p",
        response_schema={
            "type": "object",
            "properties": {"title": {"type": "string"}, "year": {"type": "integer"}},
            "required": ["title", "year"],
            "additionalProperties": False,
        },
        match="fields",
        match_fields=["title", "year"],
        expected={"title": "Dune", "year": 1984},
    )
    base.update(over)
    return Case(**base)


def test_score_passes_on_matching_fields():
    assert score(_movie_case(), {"title": "Dune", "year": 1984}).ok


def test_score_fails_on_wrong_field():
    r = score(_movie_case(), {"title": "Dune", "year": 2021})
    assert not r.ok
    assert "year" in r.detail


def test_score_fails_on_schema_violation():
    # year as a string violates the schema regardless of field matching.
    r = score(_movie_case(), {"title": "Dune", "year": "nineteen"})
    assert not r.ok
    assert "schema" in r.detail


def test_score_exact_mode():
    case = _movie_case(match="exact")
    assert score(case, {"title": "Dune", "year": 1984}).ok
    assert not score(case, {"title": "Dune", "year": 1984, "extra": 1}).ok


def test_corpus_is_self_consistent():
    # Every shipped golden case must have a valid schema and a fixture that
    # satisfies it. This is exactly the CI gate.
    results = check_corpus(load_cases())
    failures = [r for r in results if not r.ok]
    assert not failures, failures
