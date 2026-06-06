# noetica.train — fine-tune, then serve your own model

The **extend it** door. Fine-tune a local model with QLoRA, export it to Ollama,
and it immediately appears in chat (Open WebUI) **and** the structured-output
API — no cloud, no API keys, FOSS all the way (Unsloth · TRL/PEFT · llama.cpp ·
Ollama).

```
 base model ──QLoRA──► LoRA adapter ──merge──► fp16 ──llama.cpp──► GGUF
                                                                     │ quantize
                                                                     ▼
   chat (Open WebUI) ◄── ollama create ◄── Modelfile ◄──────── q4_k_m GGUF
            ▲                  │
            └── API /v1/* ◄────┘   (same Ollama registry feeds both)
```

## Install (GPU box)

These deps are **not** in the core install. Pull them only here:

```sh
pip install ".[train]"      # torch, transformers, trl, peft, unsloth
```

`unsloth` wants an NVIDIA GPU. ~16 GB VRAM fine-tunes a 7B in 4-bit.

## 1 · Prepare + validate data

Chat JSONL, one object per line (see [`examples/data/sample_chat.jsonl`](../../examples/data/sample_chat.jsonl)):

```json
{"messages": [
  {"role": "system", "content": "You are a terse assistant."},
  {"role": "user", "content": "Capital of France?"},
  {"role": "assistant", "content": "Paris."}
]}
```

Validate before you rent a GPU (this step needs no torch):

```sh
python -m noetica.train.data validate examples/data/sample_chat.jsonl
```

## 2 · Fine-tune (QLoRA)

```sh
python -m noetica.train.finetune --config noetica/train/configs/qwen2.5-7b.yaml
```

Recipes are YAML data in [`configs/`](configs/) — edit `base_model`, `dataset`,
LoRA rank, epochs, etc. Output is a LoRA adapter under `output_dir`.

## 3 · Export to Ollama → serve + chat

```sh
python -m noetica.train.export_ollama \
  --base unsloth/Qwen2.5-7B-Instruct \
  --adapter outputs/qwen2.5-7b-noetica \
  --name my-model --quantize q4_k_m \
  --llama-cpp ../llama.cpp
```

This merges the adapter, converts to GGUF, quantizes, writes a `Modelfile`, and
runs `ollama create my-model`. After that, `my-model` is selectable in Open WebUI
and usable in the API:

```sh
curl localhost:8001/v1/llm/structured \
  -d '{"model":"my-model","prompt":"...","response_schema":{"type":"object"}}'
python -m noetica.eval.run --model my-model        # score it
```

Already have a GGUF (e.g. from Unsloth's `save_pretrained_gguf`)? Skip straight
to registration — no torch needed:

```sh
python -m noetica.train.export_ollama --gguf my-model.Q4_K_M.gguf --name my-model
```

Use `--dry-run` to print every command without touching Ollama.

## CPU vs GPU — honestly

| Path | Hardware | Notes |
|------|----------|-------|
| QLoRA fine-tune | **NVIDIA GPU** (~16 GB for 7B) | Unsloth needs CUDA. CPU is not practical. |
| Colab notebook | free/cheap cloud GPU | [`notebooks/finetune_qwen_colab.ipynb`](../../notebooks/finetune_qwen_colab.ipynb) — the no-local-GPU path. |
| Export (have GGUF) | **any** (CPU fine) | `--gguf` path only writes a Modelfile + `ollama create`. |
| Serve + chat the result | **any** | Ollama runs the quantized GGUF on CPU or GPU. |

Mac users: Apple-silicon QLoRA via Unsloth isn't supported today — fine-tune on
Colab or a Linux/NVIDIA box, then export the GGUF and serve it locally on the Mac.
