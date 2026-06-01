ARG DOCKERHUB_PREFIX=
FROM ${DOCKERHUB_PREFIX}python:3.12-slim

ARG DEBIAN_MIRROR=
ARG DEBIAN_SECURITY_MIRROR=
ARG PYPI_INDEX_URL=

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    MEMFORGE_BASE_DIR=/data \
    MEMFORGE_ADMIN_API_PORT=8765 \
    MEMFORGE_CORS_ORIGINS=http://localhost:5174,http://127.0.0.1:5174 \
    MEMFORGE_PDF_RENDERER=auto \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app

RUN if [ -n "$DEBIAN_SECURITY_MIRROR" ]; then \
        sed -i "s#URIs: http://deb.debian.org/debian-security#URIs: ${DEBIAN_SECURITY_MIRROR%/}#g" /etc/apt/sources.list.d/debian.sources; \
    fi \
    && if [ -n "$DEBIAN_MIRROR" ]; then \
        sed -i "s#URIs: http://deb.debian.org/debian#URIs: ${DEBIAN_MIRROR%/}#g" /etc/apt/sources.list.d/debian.sources; \
    fi \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        fonts-liberation \
        libharfbuzz-subset0 \
        libpango-1.0-0 \
        libpangoft2-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN if [ -n "$PYPI_INDEX_URL" ]; then \
        PIP_INDEX_URL="$PYPI_INDEX_URL" pip install --root-user-action=ignore --no-cache-dir uv==0.11.17; \
    else \
        pip install --root-user-action=ignore --no-cache-dir uv==0.11.17; \
    fi

COPY pyproject.toml uv.lock README.md LICENSE ./

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -n "$PYPI_INDEX_URL" ]; then \
        UV_DEFAULT_INDEX="$PYPI_INDEX_URL" uv sync --frozen --no-dev --no-install-project; \
    else \
        uv sync --locked --no-dev --no-install-project; \
    fi

COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -n "$PYPI_INDEX_URL" ]; then \
        UV_DEFAULT_INDEX="$PYPI_INDEX_URL" uv sync --frozen --no-dev; \
    else \
        uv sync --locked --no-dev; \
    fi

RUN useradd --create-home --uid 10001 memforge \
    && mkdir -p /data \
    && chown -R memforge:memforge /data

USER memforge

EXPOSE 8765

CMD ["memforge", "api", "--host", "0.0.0.0", "--port", "8765"]
