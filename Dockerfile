# ── SQL Analytics OpenEnv — Dockerfile ───────────────────────────────────────
# Compatible with HuggingFace Spaces (Docker SDK)
# Runtime: python:3.11-slim  |  Port: 8000  |  User: non-root (appuser)

FROM python:3.11-slim

LABEL maintainer="sql-analytics-env"
LABEL version="0.1.0"
LABEL description="SQL Analytics OpenEnv environment for RL agents"

# Keeps Python from generating .pyc files + enables unbuffered stdout logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONIOENCODING=utf-8

WORKDIR /app

# Install system dependencies (build-essential needed for some pip wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies BEFORE copying source code ───────────────────
# (Docker layer cache: re-run only when pyproject.toml changes, not source)
COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install \
        "openenv-core>=0.1.0" \
        "fastapi>=0.110.0" \
        "uvicorn[standard]>=0.29.0" \
        "pydantic>=2.0.0"

# ── Copy application source ───────────────────────────────────────────────────
COPY . .

# ── Health check (Docker-native, used by orchestrators) ──────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Non-root user (required by HuggingFace Spaces) ───────────────────────────
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
