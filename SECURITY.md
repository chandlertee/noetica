# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue for security vulnerabilities. Instead, open
a private report via GitHub:
[**Report a vulnerability**](https://github.com/chandlertee/noetica/security/advisories/new).

Include a description, reproduction steps, and the affected version/commit. We aim
to acknowledge reports within a few days.

## Supported versions

This is a young project; security fixes land on `main` and the latest tagged
release.

| Version | Supported |
|---------|-----------|
| 1.0.x   | ✅        |
| < 1.0   | ❌        |

## Threat model & safe defaults

Noetica is designed to run **locally**. Defaults assume a single-user machine on
`localhost`, not a hostile network. Nothing leaves your hardware: chat, the API,
training, and the models all run locally.

The defaults are *convenient*, not *hardened*. Before exposing any part of the
stack beyond `localhost`, work through the checklist below.

### Before exposing on a LAN or the internet

- [ ] **API auth** — set `NOETICA_API_KEY` to a long random value. When set, every
  `/v1/*` request (except `/v1/health`) requires `X-API-Key` or
  `Authorization: Bearer`. The key is compared in constant time.
- [ ] **CORS** — set `NOETICA_CORS_ORIGINS` to only the origins that need browser
  access. Don't use `*` together with an API key (the CORS spec disables
  credentials with a wildcard).
- [ ] **Chat auth** — set `WEBUI_AUTH=True` for Open WebUI and create an admin
  account. The default (`False`) has **no login wall**.
- [ ] **TLS** — terminate TLS at a reverse proxy (Caddy, Traefik, nginx). Don't
  serve the API or chat over plain HTTP across a network.
- [ ] **Ollama** — Ollama's own port (`11434`) has no auth. Keep it bound to
  `localhost` / the internal Docker network; don't publish it publicly.
- [ ] **`/metrics`** — the Prometheus endpoint is unauthenticated by design (it's
  meant for a scraper on a trusted network). Don't expose it publicly; scrape it
  over the internal network or put it behind your proxy's auth.
- [ ] **Secrets** — keep keys in `.env` (gitignored), never in the image or repo.

## What we guarantee

- No telemetry, no phone-home, no third-party cloud calls in the default stack.
- No secrets are committed; CI scans the full history with gitleaks on every push.
