"""Pydantic request/response schemas for the firewall API.

All data crossing the HTTP boundary is validated here. The app layer imports
exclusively from this module — no raw dicts in route handlers.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    """Request body for ``POST /analyze``.

    Attributes:
        prompt: The raw user prompt to analyse.
        context: Optional surrounding context (e.g. system prompt). Accepted but
            not yet used by the classification pipeline.
    """

    prompt: str = Field(..., min_length=1, max_length=8192)
    context: str | None = Field(default=None, max_length=8192)


class ClassificationScore(BaseModel):
    """Per-class probability from the classifier."""

    label: str
    score: float = Field(..., ge=0.0, le=1.0)


class AnalysisResponse(BaseModel):
    """Response body for ``POST /analyze``.

    Attributes:
        decision: Final routing decision — ``"PASS"`` or ``"BLOCK"``.
        zone: Classifier zone assignment — ``"CLEAN"``, ``"GRAY"``, or ``"BLOCK"``.
        top_label: Highest-scoring threat category.
        scores: Full per-class probability distribution.
        explanation: Human-readable rationale for the decision.
        judge_invoked: Whether the LLM judge was consulted (GRAY zone only).
    """

    decision: str
    zone: str
    top_label: str
    scores: list[ClassificationScore]
    explanation: str
    judge_invoked: bool = False


class HealthResponse(BaseModel):
    """Response body for ``GET /health``."""

    status: str = "ok"
    model_loaded: bool
