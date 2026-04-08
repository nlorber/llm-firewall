"""LangGraph node implementations for the firewall routing graph.

Each node is a function ``(state: FirewallState) -> dict`` that returns a
partial state update. Nodes are composed into the graph in
:mod:`firewall.orchestrator.graph`.

Nodes:
    classify_node  — runs DeBERTa inference and assigns a threat zone
    judge_node     — invokes the LLM judge for GRAY zone prompts
    execute_node   — finalises a PASS decision
    log_node       — records a structured BLOCK event and finalises the decision

Conditional edge helpers:
    route_after_classify  — maps zone to the next node
    route_after_judge     — maps judge decision to the next node
"""
from __future__ import annotations

from firewall.orchestrator.state import FirewallState


def classify_node(state: FirewallState) -> dict:
    """Run the fine-tuned classifier and assign a threat zone.

    Zone assignment (thresholds from ``configs/orchestrator.yaml``):
    - ``top_score < clean_threshold``  → CLEAN → pass immediately
    - ``clean_threshold ≤ top_score < block_threshold`` → GRAY → route to judge
    - ``top_score ≥ block_threshold``  → BLOCK → block immediately

    Args:
        state: Current graph state (requires ``prompt``).

    Returns:
        Partial update with ``classification`` and ``zone``.
    """
    raise NotImplementedError


def judge_node(state: FirewallState) -> dict:
    """Invoke the LLM judge for GRAY zone prompts.

    Args:
        state: Current graph state (requires ``prompt`` and ``classification``).

    Returns:
        Partial update with ``judge_result``.
    """
    raise NotImplementedError


def execute_node(state: FirewallState) -> dict:
    """Finalise a PASS decision and compose a human-readable explanation.

    Args:
        state: Current graph state.

    Returns:
        Partial update with ``final_decision = "PASS"`` and ``explanation``.
    """
    raise NotImplementedError


def log_node(state: FirewallState) -> dict:
    """Append a structured BLOCK log entry and finalise the decision.

    Args:
        state: Current graph state.

    Returns:
        Partial update with ``final_decision = "BLOCK"``, ``explanation``,
        and an appended entry in ``logs``.
    """
    raise NotImplementedError


def route_after_classify(state: FirewallState) -> str:
    """Conditional edge: select the next node based on the assigned zone.

    Args:
        state: Current graph state (requires ``zone``).

    Returns:
        One of ``"execute_node"``, ``"judge_node"``, or ``"log_node"``.
    """
    raise NotImplementedError


def route_after_judge(state: FirewallState) -> str:
    """Conditional edge: select the next node based on the judge's decision.

    Args:
        state: Current graph state (requires ``judge_result``).

    Returns:
        Either ``"execute_node"`` or ``"log_node"``.
    """
    raise NotImplementedError
