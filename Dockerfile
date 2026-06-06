# Multi-stage build for the Noetica structured-output API (the `serve` door).
# Uses uv for dependency resolution to stay aligned with the dev workflow
# (`uv sync` / `uv run`). The same pyproject.toml drives both.
#
# This image is intentionally light: no GPU, no torch. The GPU work happens in
# the `ollama` container (chat/serve) or in the separate training profile.

FROM python:3.12-slim AS build

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Install uv (single static binary, ~10MB).
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/

WORKDIR /srv

# Install deps in their own layer so source changes don't bust the cache.
COPY pyproject.toml ./
RUN uv venv /opt/venv --python 3.12 \
 && VIRTUAL_ENV=/opt/venv uv pip install --no-cache \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.30" \
        "httpx>=0.27" \
        "pydantic>=2.7" \
        "pydantic-settings>=2.4" \
        "jsonschema>=4.23" \
        "prometheus-client>=0.20"

# ----- runtime stage -----
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    NOETICA_HOST=0.0.0.0 \
    NOETICA_PORT=8001 \
    NOETICA_CACHE_DIR=/data/cache \
    OLLAMA_URL=http://ollama:11434

WORKDIR /srv

COPY --from=build /opt/venv /opt/venv
COPY noetica /srv/noetica

# Run as a non-root user — cheap defense-in-depth. Fixed uid keeps mounted
# cache-volume permissions stable across rebuilds.
RUN useradd --uid 1001 --no-create-home --shell /usr/sbin/nologin noetica \
 && mkdir -p /data/cache \
 && chown -R noetica:noetica /data
USER noetica

EXPOSE 8001

# Basic healthcheck — docker-compose surfaces this in `docker ps`.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
                   r=urllib.request.urlopen('http://127.0.0.1:8001/', timeout=3); \
                   sys.exit(0 if r.status==200 else 1)" || exit 1

CMD ["uvicorn", "noetica.serve.main:app", "--host", "0.0.0.0", "--port", "8001"]
