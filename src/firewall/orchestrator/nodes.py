# src/firewall/orchestrator/nodes.py
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from firewall.classifier.model import FirewallClassifier
    from firewall.judge.base import Judge

from firewall.judge.local_judge import LocalJudge
from firewall.judge.tiered import Tier, TieredJudge
from firewall.orchestrator.metrics import (
    classification_label_total,
    classify_duration,
    judge_duration,
    judge_tier_total,
    requests_total,
)
from firewall.orchestrator.state import (
    FirewallState,  # noqa: TCH001 — LangGraph introspects annotations
)

logger = logging.getLogger(__name__)

DEFAULT_CLEAN_THRESHOLD: float = 0.3
DEFAULT_BLOCK_THRESHOLD: float = 0.8

# Module-level state — populated once by init_nodes() at startup, then read-only.
# This is process-global by design: a single firewall process serves one classifier
# + judge, and FastAPI's lifespan calls init_nodes() exactly once before any request.
# Because the globals are only written at init and read thereafter, concurrent request
# handling within the process is safe. It is NOT designed for hosting multiple graphs
# with different classifiers in one process — that would require instance-scoped state.
_classifier: FirewallClassifier | None = None
_judge: Judge | None = None
_clean_threshold: float = DEFAULT_CLEAN_THRESHOLD
_block_threshold: float = DEFAULT_BLOCK_THRESHOLD


def init_nodes(
    classifier: FirewallClassifier,
    judge: Judge,
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
    if _classifier is None:
        raise RuntimeError("call init_nodes() before using the graph")
    start = time.perf_counter()
    results = _classifier.predict([state["prompt"]])
    elapsed = time.perf_counter() - start
    scores = results[0]
    top_label = max(scores, key=scores.__getitem__)
    top_score = scores[top_label]

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
            "label": top_label,
            "scores": scores,
            "top_score": top_score,
            "threat_score": threat_score,
        },
        "zone": zone,
    }


def judge_node(state: FirewallState) -> dict[str, Any]:
    """Invoke LLM judge for GRAY zone prompts."""
    if _judge is None:
        raise RuntimeError("call init_nodes() before using the graph")
    clf = state["classification"]
    if clf is None:
        raise RuntimeError("classify_node must run before judge_node")
    start = time.perf_counter()
    # A TieredJudge carries tier provenance via decide(); a plain LocalJudge resolves
    # entirely on-device, so we record tier="local" for it too — otherwise the demo UI
    # falls back to attributing the on-device verdict to the Claude judge.
    tier: str | None = None
    reason: str | None = None
    if isinstance(_judge, TieredJudge):
        outcome = _judge.decide(
            prompt=state["prompt"],
            classification_label=clf["label"],
            scores=clf["scores"],
        )
        verdict = outcome.verdict
        tier = str(outcome.tier)
        reason = str(outcome.reason)
    else:
        verdict = _judge.judge(
            prompt=state["prompt"],
            classification_label=clf["label"],
            scores=clf["scores"],
        )
        if isinstance(_judge, LocalJudge):
            tier = str(Tier.LOCAL)
    if tier is not None:
        judge_tier_total.labels(tier=tier).inc()
    elapsed = time.perf_counter() - start
    judge_duration.labels(decision=verdict.decision).observe(elapsed)
    return {
        "judge_result": {
            "decision": verdict.decision,
            "reasoning": verdict.reasoning,
            "confidence": verdict.confidence,
            "tier": tier,
            "reason": reason,
        }
    }


def execute_node(state: FirewallState) -> dict[str, Any]:
    """Finalise a PASS decision."""
    clf: dict[str, Any] = dict(state.get("classification") or {})
    judge_result = state.get("judge_result")

    if judge_result:
        # Gray-zone prompt the judge cleared — attribute the call to the judge, not the
        # threshold (the threat score was in the gray band; the threshold did not decide).
        explanation = f"LLM judge decision: PASS. Reasoning: {judge_result['reasoning']}"
    else:
        label = clf.get("label", "unknown")
        score = clf.get("threat_score", 0.0)
        explanation = f"Prompt classified as '{label}' (threat_score {score:.2f}) — below block threshold. PASS."

    requests_total.labels(zone=state.get("zone", "unknown"), final_decision="PASS").inc()
    return {"final_decision": "PASS", "explanation": explanation}


def log_node(state: FirewallState) -> dict[str, Any]:
    """Record the block event and finalise a BLOCK decision.

    The audit line carries decision metadata only — never the prompt text, which is
    attacker-controlled and (for the local/tiered backends) the very thing that never leaves
    the machine. Length is enough to correlate with an upstream access log.
    """
    clf: dict[str, Any] = dict(state.get("classification") or {})
    judge_result = state.get("judge_result")

    if judge_result:
        explanation = f"LLM judge decision: BLOCK. Reasoning: {judge_result['reasoning']}"
    else:
        label = clf.get("label", "unknown")
        score = clf.get("threat_score", 0.0)
        explanation = f"Prompt classified as '{label}' (threat_score {score:.2f}) — above block threshold. BLOCK."

    logger.info(
        "BLOCK zone=%s label=%s threat=%.3f tier=%s prompt_len=%d",
        state.get("zone"),
        clf.get("label"),
        clf.get("threat_score", 0.0),
        judge_result.get("tier") if judge_result else None,
        len(state["prompt"]),
    )
    requests_total.labels(zone=state.get("zone", "unknown"), final_decision="BLOCK").inc()
    return {"final_decision": "BLOCK", "explanation": explanation}


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
    if judge_result is None:
        raise RuntimeError("judge_node must run before route_after_judge")
    decision = judge_result["decision"]
    return "execute_node" if decision == "PASS" else "log_node"
