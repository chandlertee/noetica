# Agentic coding over your repo — a local model that edits files

The chat door ([`chat_projects.md`](chat_projects.md)) is great for *asking about*
your code, but it's document RAG — it retrieves snippets, it doesn't change
anything. For an *agentic* loop — a tool that reads your repo, edits files, runs
commands, and commits — point [**Aider**](https://aider.chat) at the same local
Ollama registry. No cloud, no keys; it talks straight to Ollama, not the serve API.

> This is the spirit of the planned `agents/` door (see [ROADMAP.md](../ROADMAP.md)):
> tools that *consume* the local model registry. Aider is the simplest one to
> start with today.

## Reality check (read this first)

Agentic coding leans hard on the model — reliable edits, staying on task across
steps. With a **7B** local model you'll do well on scoped, single-/few-file
edits and Q&A; expect friction on large multi-file refactors. It gets genuinely
strong around `qwen2.5-coder:32b`, which needs ~64 GB of unified memory.

| Unified memory | Model | Notes |
|---|---|---|
| 8 GB | `qwen2.5-coder:3b` | Usable, modest |
| 16 GB | `qwen2.5-coder:7b` | **Best default** — the sweet spot |
| 32 GB | `qwen2.5-coder:14b` | Noticeably sharper |
| 64 GB+ | `qwen2.5-coder:32b` | Near-SOTA local coding |

Aider uses a **diff/edit-block** format rather than function-calling, so it
degrades the most gracefully on smaller local models — that's why it's the
recommended starting point here.

## 1 · Install Aider

```sh
# uv (already in the Noetica toolchain) installs it as an isolated CLI tool:
uv tool install aider-chat
# or: pipx install aider-chat   /   python -m pip install aider-install && aider-install

aider --version
```

## 2 · Pull a coding model

```sh
ollama pull qwen2.5-coder:7b      # ~4.7 GB; match the size to your RAM (table above)
```

## 3 · Point Aider at local Ollama

Two gotchas dominate local setups, both handled by config files in your home dir:

1. **The endpoint** — Aider reaches Ollama via `OLLAMA_API_BASE`.
2. **The context window** — Ollama defaults to a tiny **2048**-token context that
   starves an agentic loop. Aider forwards `num_ctx` to Ollama, so set it.

`~/.aider.conf.yml`:

```yaml
model: ollama_chat/qwen2.5-coder:7b   # note: ollama_chat/ (chat endpoint), not ollama/
set-env:
  - OLLAMA_API_BASE=http://127.0.0.1:11434
# Keep commits under your own git identity:
attribute-author: false
attribute-committer: false
```

`~/.aider.model.settings.yml`:

```yaml
- name: ollama_chat/qwen2.5-coder:7b
  extra_params:
    num_ctx: 16384   # 8192 if RAM is tight; 32768 if you can spare it
```

## 4 · Run it on a project

```sh
cd /path/to/your/project        # must be a git repo (aider commits as it goes)
aider                           # uses the config above

# or override per-run:
aider --model ollama_chat/qwen2.5-coder:7b
```

Then, inside the Aider prompt:

- `/add path/to/file.py` — put specific files in the editable context. Be
  selective: small local models work far better with a tight, relevant context
  than with the whole repo.
- Describe a change in plain English → Aider proposes edits, applies them, and
  makes a git commit you can `git diff`/revert.
- `/architect` — a two-step "plan, then edit" mode that noticeably improves
  results on weaker models.
- `/undo` reverts Aider's last commit; `/run pytest` runs a command and feeds the
  output back; `/help` lists everything.

## Other open-source options

All of these also talk to your local Ollama — pick by where you want to work:

| Tool | Surface | Notes |
|---|---|---|
| [Continue.dev](https://continue.dev) | VS Code / JetBrains | Chat + autocomplete + agent mode; mature Ollama support |
| [Cline](https://github.com/cline/cline) / [Roo Code](https://github.com/RooCodeInc/Roo-Code) | VS Code | Plan→act agent that edits files and runs commands |
| [OpenHands](https://github.com/All-Hands-AI/OpenHands) | Docker sandbox | Fuller autonomy for multi-step tasks; heavier |
| [Goose](https://github.com/block/goose) | CLI | Extensible, MCP-based tools |

## Security note

Everything here is local — Aider's traffic goes to `127.0.0.1:11434` (Ollama) and
nothing leaves the box. Aider executes file edits and (with your confirmation)
shell commands in whatever repo you launch it in, so run it on code you trust and
review its diffs before pushing. Same posture as the rest of Noetica: see
[SECURITY.md](../SECURITY.md).
