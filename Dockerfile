# syntax=docker/dockerfile:1.7

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    MEMFORGE_BASE_DIR=/data \
    MEMFORGE_ADMIN_API_PORT=8765 \
    MEMFORGE_CORS_ORIGINS=http://localhost:5174,http://127.0.0.1:5174 \
    MEMFORGE_CHROME_PATH=/usr/bin/chromium \
    MEMFORGE_CHROME_NO_SANDBOX=1 \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates chromium fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.11.17

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

RUN useradd --create-home --uid 10001 memforge \
    && mkdir -p /data \
    && chown -R memforge:memforge /data

USER memforge

EXPOSE 8765

CMD ["memforge", "api", "--host", "0.0.0.0", "--port", "8765"]
