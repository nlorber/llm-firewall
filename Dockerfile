# ── Builder stage: install all runtime dependencies ───────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN pip install --upgrade pip

# Copy only what pip needs to resolve and install deps
COPY pyproject.toml ./
COPY src/ ./src/

# Install project + runtime deps (no dev extras)
RUN pip install --no-cache-dir .

# ── Runtime stage: lean image with only what's needed to run ──────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source and config
COPY src/     ./src/
COPY configs/ ./configs/

EXPOSE 8000

# Inject the path to the fine-tuned checkpoint at runtime via env var
ENV MODEL_PATH=/app/models/classifier

CMD ["uvicorn", "firewall.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
