# The full loop: fine-tune → export → serve → chat → evaluate

End to end, this turns a base model into *your* model and makes it instantly
usable by all three front doors. FOSS throughout (Unsloth · TRL/PEFT · llama.cpp
· Ollama). No cloud inference.

```
   data ──► QLoRA fine-tune ──► export to Ollama ──► appears in chat + API ──► eval
  (you)      (GPU/Colab)         (Modelfile)          (Open WebUI + /v1/*)    (scores)
```

## 1 · Validate your dataset (no GPU)

Chat JSONL — one object per line. Start from
[`data/sample_chat.jsonl`](data/sample_chat.jsonl):

```sh
python -m noetica.train.data validate examples/data/sample_chat.jsonl
# 6/6 records valid ✓
```

The validator reports the line number of any bad record (wrong role, empty
content, no assistant turn, malformed JSON).

## 2 · Fine-tune with QLoRA

On an NVIDIA GPU (~16 GB for a 7B):

```sh
pip install ".[train]"
python -m noetica.train.finetune --config noetica/train/configs/qwen2.5-7b.yaml
# → outputs/qwen2.5-7b-noetica/  (a LoRA adapter)
```

No local GPU? Run [`notebooks/finetune_qwen_colab.ipynb`](../notebooks/finetune_qwen_colab.ipynb)
on a free Colab T4 and download the GGUF it produces. Training borrows a GPU;
inference stays local.

## 3 · Export to Ollama

**From a base + adapter** (needs llama.cpp checked out for GGUF conversion):

```sh
python -m noetica.train.export_ollama \
  --base unsloth/Qwen2.5-7B-Instruct \
  --adapter outputs/qwen2.5-7b-noetica \
  --name noetica-qwen --quantize q4_k_m \
  --llama-cpp ../llama.cpp
```

**From a GGUF you already have** (e.g. the Colab output) — no torch needed:

```sh
python -m noetica.train.export_ollama --gguf unsloth.Q4_K_M.gguf --name noetica-qwen
```

Either way it writes a `Modelfile` and runs `ollama create noetica-qwen`. Add
`--dry-run` to preview every command first.

## 4 · It's now in chat AND the API — same registry

No restart, no config change — both doors read the same Ollama registry:

```sh
ollama list | grep noetica-qwen            # registered

# API:
curl localhost:8001/v1/llm/structured -d '{
  "model": "noetica-qwen",
  "prompt": "Is my data sent to the cloud?",
  "response_schema": {"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}
}'
```

In **Open WebUI** (<http://localhost:3000>), open the model picker — `noetica-qwen`
is there. Chat with it, or attach it to a preset over your documents
(see [chat_projects.md](chat_projects.md)).

## 5 · Evaluate it

Score your fine-tune against the golden cases and compare it to the base model:

```sh
python -m noetica.eval.run \
  --model noetica-qwen \
  --model qwen2.5:7b-instruct
```

```
case                      noetica-qwen  qwen2.5:7b-instruct
-----------------------------------------------------------
contact_extraction        PASS          PASS
movie_extraction          PASS          PASS
sentiment_classification  PASS          PASS
-----------------------------------------------------------
6/6 checks passed
```

That's the loop closed: your data shaped a model, the model is served and
chattable, and you have numbers to decide whether it's better. Iterate.
