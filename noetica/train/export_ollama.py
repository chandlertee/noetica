"""Export a fine-tuned model to Ollama so it's instantly servable + chattable.

The closing leg of the loop. Two entry points:

1. **Have a GGUF already?** (e.g. Unsloth's ``save_pretrained_gguf``)::

       python -m noetica.train.export_ollama --gguf model.Q4_K_M.gguf --name my-model

   No torch needed: writes a Modelfile and runs ``ollama create``. The model
   then shows up in chat (Open WebUI) and the API automatically.

2. **Have a base + LoRA adapter?** Run the full pipeline::

       python -m noetica.train.export_ollama \
         --base unsloth/Qwen2.5-7B-Instruct \
         --adapter outputs/qwen2.5-7b-noetica \
         --name my-model --quantize q4_k_m \
         --llama-cpp ../llama.cpp

   merge adapter → fp16 HF model → GGUF (llama.cpp) → quantize → Modelfile →
   ``ollama create``.

Only the merge step needs the ``[train]`` extra (torch/peft); GGUF conversion
and quantization shell out to a local llama.cpp checkout, and registration shells
out to ``ollama``. ``--dry-run`` prints the plan without touching Ollama.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_SYSTEM = None  # inherit the base model's template/system unless overridden


def build_modelfile(
    gguf_path: str | Path,
    *,
    system: str | None = None,
    parameters: dict[str, object] | None = None,
    template: str | None = None,
) -> str:
    """Render an Ollama Modelfile referencing a local GGUF. Pure string builder."""
    lines = [f"FROM {gguf_path}"]
    for key, value in (parameters or {}).items():
        lines.append(f"PARAMETER {key} {value}")
    if template is not None:
        lines.append(f'TEMPLATE """{template}"""')
    if system is not None:
        lines.append(f'SYSTEM """{system}"""')
    return "\n".join(lines) + "\n"


def _run(cmd: list[str], *, dry_run: bool, cwd: str | None = None) -> None:
    printable = " ".join(cmd)
    print(f"$ {printable}")
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=cwd)


def merge_adapter(base: str, adapter: str, out_dir: Path, *, dry_run: bool) -> Path:
    """Merge a LoRA adapter into its base model → an fp16 HF model directory."""
    print(f"→ merging adapter {adapter} into {base}")
    if dry_run:
        print("   (dry-run: skipping torch merge)")
        return out_dir
    # Lazy heavy imports — only this path needs them.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    return out_dir


def convert_to_gguf(model_dir: Path, out_gguf: Path, llama_cpp: Path, *, dry_run: bool) -> Path:
    """fp16 HF model dir → f16 GGUF via llama.cpp's converter."""
    converter = llama_cpp / "convert_hf_to_gguf.py"
    _run(
        [
            sys.executable,
            str(converter),
            str(model_dir),
            "--outfile",
            str(out_gguf),
            "--outtype",
            "f16",
        ],
        dry_run=dry_run,
    )
    return out_gguf


def quantize(in_gguf: Path, out_gguf: Path, qtype: str, llama_cpp: Path, *, dry_run: bool) -> Path:
    """Quantize a GGUF (e.g. q4_k_m) with llama.cpp's llama-quantize."""
    quantizer = shutil.which("llama-quantize") or str(llama_cpp / "llama-quantize")
    _run([quantizer, str(in_gguf), str(out_gguf), qtype], dry_run=dry_run)
    return out_gguf


def ollama_create(name: str, modelfile: Path, *, dry_run: bool) -> None:
    _run(["ollama", "create", name, "-f", str(modelfile)], dry_run=dry_run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="noetica.train.export_ollama", description=__doc__)
    parser.add_argument("--name", required=True, help="Ollama model name to create.")
    parser.add_argument("--gguf", help="Path to an existing GGUF (skips merge/convert).")
    parser.add_argument("--base", help="Base model (HF id or path) for the merge path.")
    parser.add_argument("--adapter", help="LoRA adapter directory for the merge path.")
    parser.add_argument(
        "--quantize", dest="qtype", default="q4_k_m", help="Quant type (default q4_k_m)."
    )
    parser.add_argument("--llama-cpp", default="../llama.cpp", help="Path to a llama.cpp checkout.")
    parser.add_argument("--out", default="outputs/export", help="Working dir for artifacts.")
    parser.add_argument(
        "--system", default=DEFAULT_SYSTEM, help="Optional SYSTEM prompt for the Modelfile."
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="Optional default temperature."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan; don't run ollama.")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    llama_cpp = Path(args.llama_cpp)

    if args.gguf:
        gguf = Path(args.gguf)
    else:
        if not (args.base and args.adapter):
            parser.error("provide --gguf, or both --base and --adapter")
        merged = merge_adapter(args.base, args.adapter, out / "merged", dry_run=args.dry_run)
        f16 = convert_to_gguf(
            merged, out / f"{args.name}.f16.gguf", llama_cpp, dry_run=args.dry_run
        )
        gguf = quantize(
            f16, out / f"{args.name}.{args.qtype}.gguf", args.qtype, llama_cpp, dry_run=args.dry_run
        )

    parameters = {"temperature": args.temperature} if args.temperature is not None else None
    modelfile_text = build_modelfile(gguf, system=args.system, parameters=parameters)
    modelfile_path = out / "Modelfile"
    modelfile_path.write_text(modelfile_text, encoding="utf-8")
    print(f"→ wrote {modelfile_path}:\n{modelfile_text}")

    ollama_create(args.name, modelfile_path, dry_run=args.dry_run)
    print(
        f"✓ created Ollama model '{args.name}'. It now shows up in chat (Open WebUI) "
        f"and the API. Try it:\n"
        f'    curl localhost:8001/v1/llm/structured -d \'{{"model":"{args.name}", ...}}\'\n'
        f"    python -m noetica.eval.run --model {args.name}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
