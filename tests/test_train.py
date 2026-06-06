"""Tests for the torch-free parts of noetica.train: data validation, config,
and the Ollama Modelfile builder. These must import without the [train] extra.
"""

from __future__ import annotations

from pathlib import Path

from noetica.train.config import TrainConfig
from noetica.train.data import train_val_split, validate_file
from noetica.train.export_ollama import build_modelfile

GOOD = (
    '{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}\n'
)


def test_validate_accepts_good_record(tmp_path: Path):
    f = tmp_path / "d.jsonl"
    f.write_text(GOOD)
    report = validate_file(f)
    assert report.ok
    assert report.total == 1 and report.valid == 1


def test_validate_flags_bad_role_with_line_number(tmp_path: Path):
    f = tmp_path / "d.jsonl"
    f.write_text(
        GOOD + '{"messages": [{"role": "wizard", "content": "x"}, '
        '{"role": "assistant", "content": "y"}]}\n'
    )
    report = validate_file(f)
    assert not report.ok
    assert any("line 2" in e and "role" in e for e in report.errors)


def test_validate_requires_assistant_ending(tmp_path: Path):
    f = tmp_path / "d.jsonl"
    f.write_text('{"messages": [{"role": "user", "content": "only a question"}]}\n')
    report = validate_file(f)
    assert not report.ok


def test_validate_reports_bad_json(tmp_path: Path):
    f = tmp_path / "d.jsonl"
    f.write_text("{ not json\n")
    report = validate_file(f)
    assert not report.ok
    assert "invalid JSON" in report.errors[0]


def test_train_val_split_is_deterministic_tail():
    records = [{"i": i} for i in range(10)]
    train, val = train_val_split(records, 0.2)
    assert len(train) == 8 and len(val) == 2
    assert val == [{"i": 8}, {"i": 9}]


def test_config_rejects_unknown_keys():
    try:
        TrainConfig.from_dict({"bogus_key": 1})
    except ValueError as exc:
        assert "bogus_key" in str(exc)
    else:
        raise AssertionError("expected ValueError on unknown key")


def test_build_modelfile_minimal():
    mf = build_modelfile("model.gguf")
    assert mf.strip() == "FROM model.gguf"


def test_build_modelfile_with_system_and_params():
    mf = build_modelfile("model.gguf", system="You are terse.", parameters={"temperature": 0.7})
    assert "FROM model.gguf" in mf
    assert "PARAMETER temperature 0.7" in mf
    assert 'SYSTEM """You are terse."""' in mf


def test_shipped_sample_dataset_is_valid():
    # The dataset referenced by the example config must validate.
    report = validate_file("examples/data/sample_chat.jsonl")
    assert report.ok, report.summary()
