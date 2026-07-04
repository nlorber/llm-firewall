# src/firewall/api/app.py
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from firewall import __version__
from firewall.api.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    ClassificationScore,
    HealthResponse,
)
from firewall.classifier.model import load_classifier
from firewall.judge.tiered import make_judge
from firewall.orchestrator.graph import build_graph
from firewall.orchestrator.metrics import REGISTRY

logger = logging.getLogger(__name__)

_SERVING_CONFIG_PATH = Path("configs/serving.yaml")
_ORCHESTRATOR_CONFIG_PATH = Path("configs/orchestrator.yaml")
_STATIC_DIR = Path(__file__).parent / "static"


def _load_config() -> dict[str, Any]:
    """Load orchestrator + serving config. Separated for easy mocking in tests."""
    serving = yaml.safe_load(_SERVING_CONFIG_PATH.read_text())
    orch = yaml.safe_load(_ORCHESTRATOR_CONFIG_PATH.read_text())
    return {
        "model_path": os.environ.get("MODEL_PATH", serving["model_path"]),
        "max_length": serving["max_length"],
        "clean_threshold": orch["clean_threshold"],
        "block_threshold": orch["block_threshold"],
        "judge_model": orch["judge_model"],
        "judge_max_tokens": orch["judge_max_tokens"],
        "judge_timeout": orch["judge_timeout"],
        "retry_count": orch["retry_count"],
        # Judge backend (default "claude" — backward-compatible). local/tiered add the SLM.
        "judge_backend": orch.get("judge_backend", "claude"),
        "local_judge_model": orch.get("local_judge_model"),
        "local_judge_adapter_path": orch.get("local_judge_adapter_path"),
        "local_judge_max_tokens": orch.get("local_judge_max_tokens", 256),
        "escalation_signal": orch.get("escalation_signal", "logprob_margin"),
        "escalation_threshold": orch.get("escalation_threshold", 0.5),
    }


def _build_judge(config: dict[str, Any]) -> Any:
    """Construct the configured judge, failing fast if a local/tiered backend can't run here."""
    backend = config["judge_backend"]
    if backend in ("local", "tiered"):
        import importlib.util

        adapter = config["local_judge_adapter_path"]
        if importlib.util.find_spec("mlx_lm") is None:
            raise RuntimeError(
                f"judge_backend {backend!r} needs mlx-lm (install the 'distill' extra)"
            )
        if not config["local_judge_model"]:
            raise RuntimeError(f"judge_backend {backend!r} needs local_judge_model in config")
        if adapter and not Path(adapter).exists():
            raise RuntimeError(f"local_judge_adapter_path not found: {adapter}")
    return make_judge(
        backend,
        teacher_model=config["judge_model"],
        teacher_max_tokens=config["judge_max_tokens"],
        retry_count=config["retry_count"],
        timeout=config["judge_timeout"],
        local_model=config["local_judge_model"],
        adapter_path=config["local_judge_adapter_path"],
        signal_mode=config["escalation_signal"],
        threshold=config["escalation_threshold"],
        max_tokens=config["local_judge_max_tokens"],
    )


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        config = _load_config()
        classifier = load_classifier(config["model_path"], max_length=config["max_length"])
        judge = _build_judge(config)
        app.state.graph = build_graph(
            classifier,
            judge,
            clean_threshold=config["clean_threshold"],
            block_threshold=config["block_threshold"],
        )
        app.state.model_loaded = True
        yield

    app = FastAPI(
        title="LLM Firewall",
        description="Prompt threat classification and routing via DeBERTa + LangGraph",
        version=__version__,
        lifespan=lifespan,
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def demo() -> HTMLResponse:
        """Serve the self-contained interactive demo console."""
        return HTMLResponse((_STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    @app.post("/analyze", response_model=AnalysisResponse)
    async def analyze(request: AnalysisRequest) -> AnalysisResponse:
        initial_state: dict[str, Any] = {
            "prompt": request.prompt,
            "classification": None,
            "zone": None,
            "judge_result": None,
            "final_decision": None,
            "explanation": None,
            "logs": [],
        }
        try:
            result = await asyncio.to_thread(app.state.graph.invoke, initial_state)
        except Exception:
            logger.exception("graph.invoke failed for prompt of length %d", len(request.prompt))
            raise HTTPException(status_code=500, detail="Internal processing error") from None
        clf = result["classification"]
        scores = [ClassificationScore(label=k, score=v) for k, v in clf["scores"].items()]
        judge_result = result["judge_result"]
        return AnalysisResponse(
            decision=result["final_decision"],
            zone=result["zone"],
            top_label=clf["label"],
            scores=scores,
            explanation=result["explanation"],
            judge_invoked=judge_result is not None,
            judge_tier=judge_result.get("tier") if judge_result else None,
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_loaded=getattr(app.state, "model_loaded", False),
        )

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(
            content=generate_latest(REGISTRY),
            media_type=CONTENT_TYPE_LATEST,
        )

    return app


# Module-level instance for uvicorn: `uvicorn firewall.api.app:app`
app = create_app()


def main() -> None:
    """Start uvicorn programmatically (reads serving config and .env)."""
    import uvicorn
    from dotenv import load_dotenv

    # Load .env (e.g. ANTHROPIC_API_KEY for the LLM judge) into the process environment
    # before the app's lifespan constructs the judge client.
    load_dotenv()

    config = yaml.safe_load(_SERVING_CONFIG_PATH.read_text())
    uvicorn.run(
        "firewall.api.app:app",
        host=config["host"],
        port=config["port"],
        log_level=config["log_level"],
    )


if __name__ == "__main__":
    main()
