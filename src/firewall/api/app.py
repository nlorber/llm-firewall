"""FastAPI application exposing the firewall pipeline as an HTTP API.

Endpoints:
    POST /analyze  — analyse a prompt and return a routing decision + explanation
    GET  /health   — liveness probe

Entry point (uvicorn)::

    uvicorn firewall.api.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations


def create_app() -> object:
    """Construct and configure the FastAPI application.

    Loads the classifier checkpoint and compiled graph from paths specified
    in ``configs/serving.yaml``. Registers the ``/analyze`` and ``/health`` routes.

    Returns:
        Configured ``fastapi.FastAPI`` instance.
    """
    raise NotImplementedError


def main() -> None:
    """CLI entry point: start uvicorn programmatically (reads serving config)."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
