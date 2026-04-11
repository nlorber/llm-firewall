# src/firewall/orchestrator/nodes.py
from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from firewall.classifier.model import FirewallClassifier
    from firewall.judge.judge import LLMJudge

from firewall.orchestrator.metrics import (
    classification_label_total,
    classify_duration,
    judge_duration,
    requests_total,
)
from firewall.orchestrator.state import (
    FirewallState,  # noqa: TCH001 — LangGraph introspects annotations
)

DEFAULT_CLEAN_THRESHOLD: float = 0.3
DEFAULT_BLOCK_THRESHOLD: float = 0.8

# Module-level state — populated by init_nodes() before building the graph
_classifier: FirewallClassifier | None = None
_judge: LLMJudge | None = None
_clean_threshold: float
_block_threshold: float


def init_nodes(
    classifier: FirewallClassifier,
    judge: LLMJudge,
    clean_threshold: float = DEFAULT_CLEAN_THRESHOLD,
    block_threshold: float = DEFAULT_BLOCK_THRESHOLD,
) -> None:
    """Inject shared resources before building the graph. Call once at startup."""
    global _classifier, _judge, _clean_threshold, _block_threshold
    _classifier = classifier
    _judge = judge
    _clean_threshold = clean_threshold
    _block_threshold = block_threshold


def classify_node(state: FirewallState) -> dict[str, Any]:
    """Run classifier and assign zone."""
    assert _classifier is not None, "call init_nodes() before using the graph"
    start = time.perf_counter()
    results = _classifier.predict([state["prompt"]])
    elapsed = time.perf_counter() - start
    scores = results[0]
    top_label = max(scores, key=scores.__getitem__)
    top_score = scores[top_label]
    label_names = list(scores.keys())
    label2id = {lbl: i for i, lbl in enumerate(label_names)}

    # Route on max threat-class score, not overall top score.
    # A confident benign prediction (e.g. benign=0.35) should not trigger the GRAY zone.
    threat_score = max((v for k, v in scores.items() if k != "benign"), default=0.0)

    if threat_score >= _block_threshold:
        zone = "BLOCK"
    elif threat_score >= _clean_threshold:
        zone = "GRAY"
    else:
        zone = "CLEAN"

    classify_duration.labels(zone=zone).observe(elapsed)
    classification_label_total.labels(label=top_label).inc()

    return {
        "classification": {
            "label":        top_label,
            "label_id":     label2id[top_label],
            "scores":       scores,
            "top_score":    top_score,
            "threat_score": threat_score,
        },
        "zone": zone,
    }


def judge_node(state: FirewallState) -> dict[str, Any]:
    """Invoke LLM judge for GRAY zone prompts."""
    assert _judge is not None, "call init_nodes() before using the graph"
    clf = state["classification"]
    assert clf is not None, "classify_node must run before judge_node"
    start = time.perf_counter()
    verdict = _judge.judge(
        prompt=state["prompt"],
        classification_label=clf["label"],
        scores=clf["scores"],
    )
    elapsed = time.perf_counter() - start
    judge_duration.labels(decision=verdict.decision).observe(elapsed)
    return {
        "judge_result": {
            "decision":   verdict.decision,
            "reasoning":  verdict.reasoning,
            "confidence": verdict.confidence,
        }
    }


def execute_node(state: FirewallState) -> dict[str, Any]:
    """Finalise a PASS decision."""
    clf: dict[str, Any] = dict(state.get("classification") or {})
    label = clf.get("label", "unknown")
    score = clf.get("threat_score", 0.0)
    explanation = f"Prompt classified as '{label}' (threat_score {score:.2f}) — below block threshold. PASS."
    requests_total.labels(zone=state.get("zone", "unknown"), final_decision="PASS").inc()
    return {"final_decision": "PASS", "explanation": explanation}


def log_node(state: FirewallState) -> dict[str, Any]:
    """Append a structured block event and finalise a BLOCK decision."""
    clf: dict[str, Any] = dict(state.get("classification") or {})
    judge_result = state.get("judge_result")

    if judge_result:
        explanation = (
            f"LLM judge decision: BLOCK. Reasoning: {judge_result['reasoning']}"
        )
    else:
        label = clf.get("label", "unknown")
        score = clf.get("threat_score", 0.0)
        explanation = f"Prompt classified as '{label}' (threat_score {score:.2f}) — above block threshold. BLOCK."

    log_entry: dict[str, Any] = {
        "timestamp":    datetime.now(UTC).isoformat(),
        "prompt":       state["prompt"],
        "zone":         state.get("zone"),
        "label":        clf.get("label"),
        "top_score":    clf.get("top_score"),
        "threat_score": clf.get("threat_score"),
        "judge_result": judge_result,
        "explanation":  explanation,
    }

    requests_total.labels(zone=state.get("zone", "unknown"), final_decision="BLOCK").inc()
    existing_logs: list[dict[str, Any]] = list(state.get("logs") or [])
    return {
        "final_decision": "BLOCK",
        "explanation":    explanation,
        "logs":           existing_logs + [log_entry],
    }


def route_after_classify(state: FirewallState) -> str:
    """Map zone to next node."""
    zone = state["zone"]
    if zone == "CLEAN":
        return "execute_node"
    if zone == "GRAY":
        return "judge_node"
    return "log_node"  # BLOCK


def route_after_judge(state: FirewallState) -> str:
    """Map judge decision to next node."""
    judge_result = state["judge_result"]
    assert judge_result is not None, "judge_node must run before route_after_judge"
    decision = judge_result["decision"]
    return "execute_node" if decision == "PASS" else "log_node"
