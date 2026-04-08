# src/firewall/orchestrator/graph.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from firewall.classifier.model import FirewallClassifier
    from firewall.judge.judge import LLMJudge


def build_graph(
    classifier: FirewallClassifier,
    judge: LLMJudge,
    clean_threshold: float = 0.3,
    block_threshold: float = 0.8,
) -> Any:  # langgraph compiled graph has no public type; Any is appropriate here
    """Assemble and compile the firewall LangGraph StateGraph.

    Call this once at API startup with the loaded classifier and judge.
    """
    from langgraph.graph import END, StateGraph

    import firewall.orchestrator.nodes as _nodes
    from firewall.orchestrator.nodes import (
        classify_node,
        execute_node,
        judge_node,
        log_node,
        route_after_classify,
        route_after_judge,
    )
    from firewall.orchestrator.state import FirewallState

    _nodes.init_nodes(classifier, judge, clean_threshold, block_threshold)

    graph = StateGraph(FirewallState)

    graph.add_node("classify_node", classify_node)
    graph.add_node("judge_node",    judge_node)
    graph.add_node("execute_node",  execute_node)
    graph.add_node("log_node",      log_node)

    graph.set_entry_point("classify_node")

    graph.add_conditional_edges(
        "classify_node",
        route_after_classify,
        {
            "execute_node": "execute_node",
            "judge_node":   "judge_node",
            "log_node":     "log_node",
        },
    )
    graph.add_conditional_edges(
        "judge_node",
        route_after_judge,
        {
            "execute_node": "execute_node",
            "log_node":     "log_node",
        },
    )
    graph.add_edge("execute_node", END)
    graph.add_edge("log_node",     END)

    return graph.compile()
