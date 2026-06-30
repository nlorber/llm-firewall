# src/firewall/orchestrator/graph.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from firewall.orchestrator.nodes import DEFAULT_BLOCK_THRESHOLD, DEFAULT_CLEAN_THRESHOLD

if TYPE_CHECKING:
    from firewall.classifier.model import FirewallClassifier
    from firewall.judge.base import Judge


def build_graph(
    classifier: FirewallClassifier,
    judge: Judge,
    clean_threshold: float = DEFAULT_CLEAN_THRESHOLD,
    block_threshold: float = DEFAULT_BLOCK_THRESHOLD,
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
    graph.add_node("judge_node", judge_node)
    graph.add_node("execute_node", execute_node)
    graph.add_node("log_node", log_node)

    graph.set_entry_point("classify_node")

    graph.add_conditional_edges(
        "classify_node",
        route_after_classify,
        {
            "execute_node": "execute_node",
            "judge_node": "judge_node",
            "log_node": "log_node",
        },
    )
    graph.add_conditional_edges(
        "judge_node",
        route_after_judge,
        {
            "execute_node": "execute_node",
            "log_node": "log_node",
        },
    )
    graph.add_edge("execute_node", END)
    graph.add_edge("log_node", END)

    return graph.compile()
