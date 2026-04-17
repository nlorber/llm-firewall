# ── Builder stage: install all runtime dependencies ───────────────────────────
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

WORKDIR /app

# Copy only what uv needs to resolve and install deps
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Install project + runtime deps (no dev extras), frozen to lockfile
RUN uv sync --frozen --no-dev

# ── Runtime stage: lean image with only what's needed to run ──────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy the uv-managed virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source and config
COPY src/     ./src/
COPY configs/ ./configs/

EXPOSE 8000

# Activate the venv
ENV PATH="/app/.venv/bin:$PATH"
# Inject the path to the fine-tuned checkpoint at runtime
ENV MODEL_PATH=/app/models/classifier

CMD ["firewall-serve"]
