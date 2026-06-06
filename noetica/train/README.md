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

## Run the whole loop with Docker on a GPU box (Windows + WSL2)

The `train` compose profile runs the fine-tune in a CUDA container, drops a
quantized **GGUF** into `./volumes/train`, and you register it with a local
Ollama. From a fresh Windows machine:

### 0 · One-time: GPU-in-Docker

1. **NVIDIA driver** — install the latest GeForce/Studio driver (recent drivers
   include WSL2 CUDA support). Reboot.
2. **WSL2** — in an admin PowerShell: `wsl --install -d Ubuntu`, reboot, set up
   the Ubuntu user.
3. **Docker Desktop** — install it; **Settings → General** → enable *Use the WSL 2
   based engine*; **Settings → Resources → WSL integration** → enable Ubuntu. GPU
   support is automatic with a recent driver.
4. **Verify the GPU reaches containers** (inside the Ubuntu/WSL2 shell):
   ```sh
   docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   ```
   You should see your GPU. Fix this before continuing if not (driver + Docker
   Desktop versions).

Do everything below **inside the WSL2 Ubuntu shell** — git, docker, and python
all live there cleanly.

### 1 · Code + a native, GPU-accelerated Ollama

```sh
sudo apt update && sudo apt install -y git python3 python3-venv curl
curl -fsSL https://ollama.com/install.sh | sh
# Bind to 0.0.0.0 so the chat/API containers can reach it via host.docker.internal:
OLLAMA_HOST=0.0.0.0 ollama serve &
ollama pull qwen2.5:7b-instruct nomic-embed-text

git clone https://github.com/chandlertee/noetica && cd noetica
```

### 2 · Fine-tune in the container (GPU)

```sh
# Validate the dataset first (no GPU, instant):
python3 -m pip install --user . && \
python3 -m noetica.train.data validate examples/data/sample_chat.jsonl

# Train — builds the CUDA image on first run (slow), then runs on the GPU.
# With gguf_quantization: q4_k_m in the recipe (the default), it also writes a GGUF.
docker compose --profile train run --rm noetica-train \
  python -m noetica.train.finetune --config noetica/train/configs/qwen2.5-7b.yaml

ls volumes/train/**/*.gguf          # find the GGUF it produced
```

### 3 · Register it with Ollama (host, torch-free)

```sh
python3 -m noetica.train.export_ollama \
  --gguf volumes/train/qwen2.5-7b-noetica/<model>.q4_k_m.gguf \
  --name noetica-qwen
ollama list | grep noetica-qwen
```

### 4 · Serve + chat + eval — the loop, closed

```sh
NOETICA_PROFILE=cpu ./bin/up        # chat :3000, API :8001 → your native Ollama

curl localhost:8001/v1/llm/structured \
  -d '{"model":"noetica-qwen","prompt":"Is my data sent to the cloud?","response_schema":{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}}'

python3 -m noetica.eval.run --model noetica-qwen --model qwen2.5:7b-instruct
```

Open <http://localhost:3000> — `noetica-qwen` is in the model picker. The full
loop on one box: trained in a GPU container, served + chatted + scored.

> **First-run note:** this exact GPU path is the one not covered by CI (no GPU on
> the runners). If the image build or `save_pretrained_gguf` hits a torch/CUDA
> mismatch, it's almost always the CUDA base vs the torch wheel — open an issue
> with the error and we'll pin it.

> **Linux GPU box:** same steps minus WSL2 — install the
> [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
> then `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`
> to verify, and follow steps 1–4.

## CPU vs GPU — honestly

| Path | Hardware | Notes |
|------|----------|-------|
| QLoRA fine-tune | **NVIDIA GPU** (~16 GB for 7B) | Unsloth needs CUDA. CPU is not practical. |
| Colab notebook | free/cheap cloud GPU | [`notebooks/finetune_qwen_colab.ipynb`](../../notebooks/finetune_qwen_colab.ipynb) — the no-local-GPU path. |
| Export (have GGUF) | **any** (CPU fine) | `--gguf` path only writes a Modelfile + `ollama create`. |
| Serve + chat the result | **any** | Ollama runs the quantized GGUF on CPU or GPU. |

Mac users: Apple-silicon QLoRA via Unsloth isn't supported today — fine-tune on
Colab or a Linux/NVIDIA box, then export the GGUF and serve it locally on the Mac.
