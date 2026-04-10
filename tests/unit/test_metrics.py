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
