# Roadmap

Noetica ships three front doors over one Ollama registry: **chat**, the
**serve** API, and **train + eval**. The roadmap extends that surface without
breaking the core contract (the `/v1/*` API and the lightweight, GPU-free
`noetica.serve` install).

## Next module — `agents/` (local agents & automation)

The next door is **automate it**: local agents that *consume* the existing serve
API + Ollama registry, rather than reach into them. It slots in exactly like
`train/` does today:

- its own optional extra, `pip install .[agents]`, so the core stays light;
- its own Docker compose profile (`--profile agents`);
- a pure consumer of `POST /v1/llm/structured` + `/v1/embed` and the local model
  registry — no new coupling into `serve/`.

Scope sketch (subject to change):

- a small tool-calling loop driven by the structured-output endpoint (JSON Schema
  *is* the tool signature);
- local tools only by default (filesystem, shell, HTTP) with explicit allow-lists;
- scheduled / triggered runs;
- traces that reuse the serve request-ID + Prometheus plumbing;
- eval cases for agent trajectories, scored like the structured-output goldens.

> Out of scope for v1.0.0 on purpose — v1 stays focused on the three doors.

## Other candidates

- **serve**: streaming responses; per-route rate limiting; OpenAI-compatible shim
  so existing SDKs point at Noetica unchanged.
- **train**: more recipes (Llama, Gemma, Phi); DPO/ORPO preference tuning; an
  eval-gated "promote adapter" flow.
- **eval**: latency/throughput benchmarking; larger golden corpora; HTML report.
- **chat**: documented multi-user setup behind auth + TLS; preset/prompt packs
  shipped in-repo.
- **ops**: a ready-made Grafana dashboard for the `/metrics` series.

Have a use case? Open an issue — see [CONTRIBUTING.md](CONTRIBUTING.md).
