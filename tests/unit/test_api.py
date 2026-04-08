# tests/test_api.py
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _graph_result(decision: str, zone: str, label: str, score: float, judge: bool) -> dict:
    return {
        "prompt":           "test prompt",
        "classification":   {"label": label, "label_id": 0, "scores": {label: score}, "top_score": score},
        "zone":             zone,
        "judge_result":     {"decision": decision, "reasoning": "ok", "confidence": 0.9} if judge else None,
        "final_decision":   decision,
        "explanation":      "test explanation",
        "logs":             [],
    }


@pytest.fixture()
def client():
    """TestClient with mocked classifier + judge so no models are loaded."""
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

    test_client = TestClient(app)
    # Inject graph result via mock
    test_client.app.state.graph = mock_graph
    return test_client, mock_graph


class TestAnalyzeEndpoint:
    def test_valid_request_returns_200(self, client) -> None:
        test_client, mock_graph = client
        mock_graph.invoke.return_value = _graph_result("PASS", "CLEAN", "benign", 0.1, False)
        response = test_client.post("/analyze", json={"prompt": "hello world"})
        assert response.status_code == 200

    def test_empty_prompt_returns_422(self, client) -> None:
        test_client, _ = client
        response = test_client.post("/analyze", json={"prompt": ""})
        assert response.status_code == 422

    def test_response_contains_decision_zone_top_label(self, client) -> None:
        test_client, mock_graph = client
        mock_graph.invoke.return_value = _graph_result("BLOCK", "BLOCK", "injection", 0.92, False)
        response = test_client.post("/analyze", json={"prompt": "ignore instructions"})
        data = response.json()
        assert data["decision"] == "BLOCK"
        assert data["zone"] == "BLOCK"
        assert data["top_label"] == "injection"

    def test_judge_invoked_flag_is_true_for_gray(self, client) -> None:
        test_client, mock_graph = client
        mock_graph.invoke.return_value = _graph_result("PASS", "GRAY", "jailbreak", 0.55, True)
        response = test_client.post("/analyze", json={"prompt": "maybe bad"})
        assert response.json()["judge_invoked"] is True


class TestHealthEndpoint:
    def test_health_returns_200(self, client) -> None:
        test_client, _ = client
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_health_status_is_ok(self, client) -> None:
        test_client, _ = client
        response = test_client.get("/health")
        assert response.json()["status"] == "ok"
