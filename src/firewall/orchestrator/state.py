"""LangGraph state schema for the firewall routing graph.

``FirewallState`` is the single source of truth for all data flowing through
the graph. Every node receives the full state and returns a partial update dict.
"""
from __future__ import annotations

from typing import Any, TypedDict


class ClassificationResult(TypedDict):
    """Output produced by the classifier node."""

    label: str
    scores: dict[str, float]   # label → probability for all classes
    top_score: float            # highest probability across all labels (including benign)
    threat_score: float         # max non-benign probability (used for zone assignment)


class JudgeResult(TypedDict):
    """Output produced by the LLM judge node (only populated for GRAY zone prompts)."""

    decision: str      # "PASS" | "BLOCK"
    reasoning: str
    confidence: float  # judge's self-reported confidence in [0, 1]


class FirewallState(TypedDict):
    """Mutable state of the LangGraph firewall graph.

    Fields are populated progressively as the prompt moves through nodes.
    Unset fields are ``None`` until the relevant node runs.
    """

    prompt: str
    classification: ClassificationResult | None
    zone: str | None              # "CLEAN" | "GRAY" | "BLOCK"
    judge_result: JudgeResult | None
    final_decision: str | None    # "PASS" | "BLOCK"
    explanation: str | None       # human-readable rationale included in the API response
    logs: list[dict[str, Any]]    # structured audit log of block events
