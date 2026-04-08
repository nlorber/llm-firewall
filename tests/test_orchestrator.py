# tests/test_orchestrator.py
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import firewall.orchestrator.nodes as nodes_mod
from firewall.orchestrator.state import FirewallState


def _make_state(prompt: str = "hello") -> FirewallState:
    return FirewallState(
        prompt=prompt,
        classification=None,
        zone=None,
        judge_result=None,
        final_decision=None,
        explanation=None,
        logs=[],
    )


def _mock_classifier(top_label: str, top_score: float) -> MagicMock:
    clf = MagicMock()
    scores = {"benign": 0.1, "injection": 0.1, "jailbreak": 0.1, "exfiltration": 0.1, "escalation": 0.1}
    scores[top_label] = top_score
    clf.predict.return_value = [scores]
    return clf


@pytest.fixture(autouse=True)
def reset_nodes():
    """Reset module-level state before each test."""
    nodes_mod.init_nodes(
        classifier=_mock_classifier("benign", 0.9),
        judge=MagicMock(),
        clean_threshold=0.3,
        block_threshold=0.8,
    )
    yield


class TestClassifyNode:
    def test_classify_node_sets_classification_and_zone(self) -> None:
        nodes_mod.init_nodes(
            classifier=_mock_classifier("benign", 0.1),   # below clean threshold → CLEAN
            judge=MagicMock(), clean_threshold=0.3, block_threshold=0.8,
        )
        update = nodes_mod.classify_node(_make_state("hello"))
        assert update["classification"]["label"] == "benign"
        assert update["zone"] == "CLEAN"

    def test_high_score_sets_block_zone(self) -> None:
        nodes_mod.init_nodes(
            classifier=_mock_classifier("injection", 0.95),
            judge=MagicMock(), clean_threshold=0.3, block_threshold=0.8,
        )
        update = nodes_mod.classify_node(_make_state("ignore prev"))
        assert update["zone"] == "BLOCK"

    def test_mid_score_sets_gray_zone(self) -> None:
        nodes_mod.init_nodes(
            classifier=_mock_classifier("jailbreak", 0.55),
            judge=MagicMock(), clean_threshold=0.3, block_threshold=0.8,
        )
        update = nodes_mod.classify_node(_make_state("maybe bad"))
        assert update["zone"] == "GRAY"


class TestRouting:
    def test_route_after_classify_clean_returns_execute(self) -> None:
        state = {**_make_state(), "zone": "CLEAN"}
        assert nodes_mod.route_after_classify(state) == "execute_node"

    def test_route_after_classify_gray_returns_judge(self) -> None:
        state = {**_make_state(), "zone": "GRAY"}
        assert nodes_mod.route_after_classify(state) == "judge_node"

    def test_route_after_classify_block_returns_log(self) -> None:
        state = {**_make_state(), "zone": "BLOCK"}
        assert nodes_mod.route_after_classify(state) == "log_node"

    def test_route_after_judge_pass_returns_execute(self) -> None:
        state = {**_make_state(), "judge_result": {"decision": "PASS", "reasoning": "", "confidence": 0.9}}
        assert nodes_mod.route_after_judge(state) == "execute_node"

    def test_route_after_judge_block_returns_log(self) -> None:
        state = {**_make_state(), "judge_result": {"decision": "BLOCK", "reasoning": "", "confidence": 0.9}}
        assert nodes_mod.route_after_judge(state) == "log_node"


class TestExecuteNode:
    def test_execute_node_sets_pass_decision(self) -> None:
        state = _make_state()
        update = nodes_mod.execute_node(state)
        assert update["final_decision"] == "PASS"
        assert update["explanation"]

    def test_execute_node_sets_explanation(self) -> None:
        state = _make_state()
        update = nodes_mod.execute_node(state)
        assert isinstance(update["explanation"], str)
        assert len(update["explanation"]) > 0


class TestLogNode:
    def test_log_node_sets_block_decision(self) -> None:
        state = _make_state("malicious prompt")
        update = nodes_mod.log_node(state)
        assert update["final_decision"] == "BLOCK"

    def test_log_node_appends_to_logs(self) -> None:
        state = _make_state("bad prompt")
        update = nodes_mod.log_node(state)
        assert len(update["logs"]) == 1
        assert "prompt" in update["logs"][0]
        assert "timestamp" in update["logs"][0]
