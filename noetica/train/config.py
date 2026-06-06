"""Training configuration.

A small dataclass loaded from YAML so recipes live as data, not code. Importing
this module does **not** pull torch — only :func:`TrainConfig.from_yaml` needs
PyYAML (shipped with the ``[train]`` extra).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    # Model
    base_model: str = "unsloth/Qwen2.5-7B-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = True  # QLoRA

    # Data
    dataset: str = "examples/data/sample_chat.jsonl"
    val_fraction: float = 0.1

    # LoRA adapter
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Optimisation
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    seed: int = 3407

    # Output
    output_dir: str = "outputs/qwen2.5-7b-noetica"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        known = {f for f in cls.__dataclass_fields__}  # noqa: C416
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        import yaml  # lazy — only needed when actually loading a recipe

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)
