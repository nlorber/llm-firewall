# src/firewall/api/app.py
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI

from firewall.api.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    ClassificationScore,
    HealthResponse,
)
from firewall.classifier.model import load_classifier
from firewall.judge.judge import LLMJudge
from firewall.orchestrator.graph import build_graph


def _load_config() -> dict[str, Any]:
    """Load orchestrator + serving config. Separated for easy mocking in tests."""
    serving = yaml.safe_load(Path("configs/serving.yaml").read_text())
    orch    = yaml.safe_load(Path("configs/orchestrator.yaml").read_text())
    return {
        "model_path":       os.environ.get("MODEL_PATH", serving["model_path"]),
        "clean_threshold":  orch["clean_threshold"],
        "block_threshold":  orch["block_threshold"],
        "judge_model":      orch["judge_model"],
        "judge_max_tokens": orch["judge_max_tokens"],
        "retry_count":      orch["retry_count"],
    }


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[type-arg]
        config = _load_config()
        classifier = load_classifier(config["model_path"])
        judge = LLMJudge(
            model=config["judge_model"],
            max_tokens=config["judge_max_tokens"],
            retry_count=config["retry_count"],
        )
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
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.post("/analyze", response_model=AnalysisResponse)
    async def analyze(request: AnalysisRequest) -> AnalysisResponse:
        initial_state = {
            "prompt":         request.prompt,
            "classification": None,
            "zone":           None,
            "judge_result":   None,
            "final_decision": None,
            "explanation":    None,
            "logs":           [],
        }
        result = app.state.graph.invoke(initial_state)
        clf = result["classification"]
        scores = [
            ClassificationScore(label=k, score=v)
            for k, v in clf["scores"].items()
        ]
        return AnalysisResponse(
            decision=result["final_decision"],
            zone=result["zone"],
            top_label=clf["label"],
            scores=scores,
            explanation=result["explanation"],
            judge_invoked=result["judge_result"] is not None,
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_loaded=getattr(app.state, "model_loaded", False),
        )

    return app


# Module-level instance for uvicorn: `uvicorn firewall.api.app:app`
app = create_app()


def main() -> None:
    """Start uvicorn programmatically (reads serving config)."""
    import uvicorn

    config = yaml.safe_load(Path("configs/serving.yaml").read_text())
    uvicorn.run(
        "firewall.api.app:app",
        host=config["host"],
        port=config["port"],
        log_level=config["log_level"],
    )


if __name__ == "__main__":
    main()
