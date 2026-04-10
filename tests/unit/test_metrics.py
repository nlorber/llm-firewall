# tests/test_metrics.py
from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import firewall.orchestrator.nodes as nodes_mod
from firewall.judge.judge import JudgeVerdict
from firewall.orchestrator.metrics import REGISTRY
from firewall.orchestrator.state import FirewallState


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


class TestJudgeNodeMetrics:
    def test_judge_observes_duration_histogram(self) -> None:
        mock_judge = MagicMock()
        mock_judge.judge.return_value = JudgeVerdict(
            decision="BLOCK", reasoning="suspicious", confidence=0.85,
        )
        nodes_mod.init_nodes(
            classifier=_mock_classifier("benign", 0.1),
            judge=mock_judge, clean_threshold=0.3, block_threshold=0.8,
        )
        state = {
            **_make_state("test"),
            "classification": {"label": "jailbreak", "scores": {"jailbreak": 0.55}},
        }
        before = _metric_value("firewall_judge_duration_seconds_count", {"decision": "BLOCK"})
        nodes_mod.judge_node(state)
        after = _metric_value("firewall_judge_duration_seconds_count", {"decision": "BLOCK"})
        assert after == before + 1


class TestTerminalNodeMetrics:
    def test_execute_node_increments_requests_total(self) -> None:
        nodes_mod.init_nodes(
            classifier=_mock_classifier("benign", 0.9),
            judge=MagicMock(), clean_threshold=0.3, block_threshold=0.8,
        )
        state = {**_make_state("hello"), "zone": "CLEAN", "classification": {"label": "benign", "top_score": 0.9}}
        before = _metric_value("firewall_requests_total", {"zone": "CLEAN", "final_decision": "PASS"})
        nodes_mod.execute_node(state)
        after = _metric_value("firewall_requests_total", {"zone": "CLEAN", "final_decision": "PASS"})
        assert after == before + 1

    def test_log_node_increments_requests_total(self) -> None:
        nodes_mod.init_nodes(
            classifier=_mock_classifier("injection", 0.95),
            judge=MagicMock(), clean_threshold=0.3, block_threshold=0.8,
        )
        state = {**_make_state("bad"), "zone": "BLOCK", "classification": {"label": "injection", "top_score": 0.95}}
        before = _metric_value("firewall_requests_total", {"zone": "BLOCK", "final_decision": "BLOCK"})
        nodes_mod.log_node(state)
        after = _metric_value("firewall_requests_total", {"zone": "BLOCK", "final_decision": "BLOCK"})
        assert after == before + 1


class TestMetricsEndpoint:
    @staticmethod
    def _make_client() -> TestClient:
        mock_clf = MagicMock()
        mock_graph = MagicMock()
        with patch("firewall.api.app.load_classifier", return_value=mock_clf), \
             patch("firewall.api.app.build_graph", return_value=mock_graph), \
             patch("firewall.api.app._load_config") as mock_cfg:
            mock_cfg.return_value = {
                "model_path": "dummy", "clean_threshold": 0.3,
                "block_threshold": 0.8, "judge_model": "dummy",
                "judge_max_tokens": 128, "retry_count": 1,
            }
            from firewall.api.app import create_app
            app = create_app()
        return TestClient(app)

    def test_metrics_returns_200(self) -> None:
        client = self._make_client()
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_content_type_is_text_plain(self) -> None:
        client = self._make_client()
        response = client.get("/metrics")
        assert "text/plain" in response.headers["content-type"]

    def test_metrics_body_contains_firewall_metrics(self) -> None:
        client = self._make_client()
        response = client.get("/metrics")
        body = response.text
        assert "firewall_classify_duration_seconds" in body
        assert "firewall_judge_duration_seconds" in body
        assert "firewall_requests" in body
        assert "firewall_classification_label" in body
