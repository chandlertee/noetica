# noetica.eval — structured-output evaluation

Golden cases that score the `/v1/llm/structured` contract across prompt and
model variants. Runs on the core install (no GPU, no torch).

## A case

Each file in [`cases/`](cases/) is one case:

```jsonc
{
  "name": "movie_extraction",
  "ci": true,                      // part of the CI regression subset
  "match": "fields",               // schema_only | fields | exact
  "match_fields": ["title", "year"],
  "prompt": "Extract movie info from: Blade Runner (1982)...",
  "response_schema": { /* JSON Schema (Draft 2020-12) */ },
  "expected": { "title": "Blade Runner", "year": 1982 },
  "fixture":  { "title": "Blade Runner", "year": 1982 }  // recorded good output
}
```

Scoring is layered: output must first **validate against the schema**, then
satisfy the `match` rule (`schema_only` stops at validation; `fields` deep-equals
the listed keys; `exact` deep-equals the whole object).

## Run it

```sh
# CI gate — validate the corpus + scorer, no model needed (deterministic):
python -m noetica.eval.run --check          # or: noetica eval --check

# Score the recorded fixtures end-to-end, still offline:
python -m noetica.eval.run --offline

# Compare live models against a running API (needs Ollama + models pulled):
noetica serve &                             # or ./bin/up
python -m noetica.eval.run \
  --api http://localhost:8001 \
  --model qwen2.5:7b-instruct \
  --model llama3.1:latest
```

The live run prints a `case × model` PASS/FAIL table and exits non-zero if any
selected case fails.

## CI regression gate

CI runs `python -m noetica.eval.run --check --ci`: it restricts to `ci: true`
cases and asserts that every shipped schema is a valid Draft 2020-12 schema and
every recorded fixture still satisfies its own case. This guards the golden
contract on every push without needing a GPU or a model in CI. Model-quality
evaluation (the live table) is run locally where the models live.

## Add a case

Drop a new `cases/<name>.json`, include a `fixture` that a good model would
produce, set `ci: true` if it's stable enough to gate on, and run `--check`.
