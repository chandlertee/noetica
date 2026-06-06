"""Unsloth QLoRA fine-tuning recipe.

Runnable on a single NVIDIA GPU (≈ 16 GB is enough for a 7B in 4-bit). The heavy
imports (unsloth/torch/trl) happen inside :func:`run` so ``--help`` and unit
tests work on a laptop without the ``[train]`` extra installed.

    python -m noetica.train.finetune --config noetica/train/configs/qwen2.5-7b.yaml

Outputs a LoRA adapter under ``config.output_dir``. Turn it into a servable
Ollama model with :mod:`noetica.train.export_ollama`.

CPU note: QLoRA on CPU is not practical. Use the Colab notebook
(``notebooks/finetune_qwen_colab.ipynb``) for the no-local-GPU path.
"""

from __future__ import annotations

import argparse
import sys

from noetica.train.config import TrainConfig
from noetica.train.data import load_records, train_val_split


def _to_text(records: list[dict], tokenizer) -> list[dict]:
    """Render each chat record to a single training string via the chat template."""
    return [
        {
            "text": tokenizer.apply_chat_template(
                r["messages"], tokenize=False, add_generation_prompt=False
            )
        }
        for r in records
    ]


def run(config: TrainConfig) -> str:
    """Fine-tune and return the adapter output directory."""
    # --- heavy imports, deferred ---
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    print(f"→ loading {config.base_model} (4bit={config.load_in_4bit})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        dtype=None,  # auto
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )

    print(f"→ loading + validating dataset {config.dataset}")
    records = load_records(config.dataset)
    train_records, val_records = train_val_split(records, config.val_fraction)
    print(f"   {len(train_records)} train / {len(val_records)} val examples")

    train_ds = Dataset.from_list(_to_text(train_records, tokenizer))
    eval_ds = Dataset.from_list(_to_text(val_records, tokenizer)) if val_records else None

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=config.max_seq_length,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_ratio=config.warmup_ratio,
            num_train_epochs=config.num_train_epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            seed=config.seed,
            output_dir=config.output_dir,
            logging_steps=1,
            optim="adamw_8bit",
            report_to="none",
        ),
    )

    print("→ training")
    trainer.train()

    print(f"→ saving LoRA adapter to {config.output_dir}")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(
        "✓ done. Next: export to Ollama —\n"
        f"    python -m noetica.train.export_ollama \\\n"
        f"      --base {config.base_model} --adapter {config.output_dir} --name my-model"
    )
    return config.output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="noetica.train.finetune", description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML training config.")
    args = parser.parse_args(argv)
    config = TrainConfig.from_yaml(args.config)
    run(config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
