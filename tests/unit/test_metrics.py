# tests/test_metrics.py
from __future__ import annotations


class TestMetricDefinitions:
    def test_classify_duration_exists(self) -> None:
        from firewall.orchestrator.metrics import classify_duration

        assert classify_duration._name == "firewall_classify_duration_seconds"
        assert "zone" in classify_duration._labelnames

    def test_judge_duration_exists(self) -> None:
        from firewall.orchestrator.metrics import judge_duration

        assert judge_duration._name == "firewall_judge_duration_seconds"
        assert "decision" in judge_duration._labelnames

    def test_requests_total_exists(self) -> None:
        from firewall.orchestrator.metrics import requests_total

        assert "firewall_requests" in requests_total._name
        assert "zone" in requests_total._labelnames
        assert "final_decision" in requests_total._labelnames

    def test_classification_label_total_exists(self) -> None:
        from firewall.orchestrator.metrics import classification_label_total

        assert "firewall_classification_label" in classification_label_total._name
        assert "label" in classification_label_total._labelnames


from unittest.mock import MagicMock

import firewall.orchestrator.nodes as nodes_mod
from firewall.orchestrator.metrics import REGISTRY
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


def _metric_value(name: str, labels: dict[str, str]) -> float:
    value = REGISTRY.get_sample_value(name, labels)
    return value if value is not None else 0.0


class TestClassifyNodeMetrics:
    def test_classify_observes_duration_histogram(self) -> None:
        nodes_mod.init_nodes(
            classifier=_mock_classifier("benign", 0.9),
            judge=MagicMock(), clean_threshold=0.3, block_threshold=0.8,
        )
        before = _metric_value("firewall_classify_duration_seconds_count", {"zone": "CLEAN"})
        nodes_mod.classify_node(_make_state("hello"))
        after = _metric_value("firewall_classify_duration_seconds_count", {"zone": "CLEAN"})
        assert after == before + 1

    def test_classify_increments_label_counter(self) -> None:
        nodes_mod.init_nodes(
            classifier=_mock_classifier("injection", 0.95),
            judge=MagicMock(), clean_threshold=0.3, block_threshold=0.8,
        )
        before = _metric_value("firewall_classification_label_total", {"label": "injection"})
        nodes_mod.classify_node(_make_state("bad"))
        after = _metric_value("firewall_classification_label_total", {"label": "injection"})
        assert after == before + 1
