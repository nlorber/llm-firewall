# tests/test_api.py
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from firewall.api.schemas import AnalysisRequest, ClassificationScore
from pydantic import ValidationError


def _graph_result(decision: str, zone: str, label: str, score: float, judge: bool) -> dict:
    return {
        "prompt": "test prompt",
        "classification": {
            "label": label,
            "scores": {label: score},
            "top_score": score,
            "threat_score": score,
        },
        "zone": zone,
        "judge_result": {"decision": decision, "reasoning": "ok", "confidence": 0.9}
        if judge
        else None,
        "final_decision": decision,
        "explanation": "test explanation",
        "logs": [],
    }


@pytest.fixture()
def client():
    """TestClient whose lifespan runs with mocked classifier, judge and graph.

    Entering ``TestClient`` as a context manager triggers the FastAPI lifespan,
    so the real startup path (config load, classifier load, judge + graph build)
    is exercised — against mocks, so no models or API keys are required.
    """
    mock_clf = MagicMock()
    mock_graph = MagicMock()

    with (
        patch("firewall.api.app.load_classifier", return_value=mock_clf),
        patch("firewall.api.app.build_graph", return_value=mock_graph),
        patch("firewall.api.app.LLMJudge", return_value=MagicMock()),
        patch("firewall.api.app._load_config") as mock_cfg,
    ):
        mock_cfg.return_value = {
            "model_path": "dummy",
            "max_length": 256,
            "clean_threshold": 0.3,
            "block_threshold": 0.8,
            "judge_model": "dummy",
            "judge_max_tokens": 128,
            "judge_timeout": 10.0,
            "retry_count": 1,
        }
        from firewall.api.app import create_app

        app = create_app()
        with TestClient(app) as test_client:
            yield test_client, mock_graph


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

    def test_graph_invoke_error_returns_sanitized_500(self, client) -> None:
        test_client, mock_graph = client
        mock_graph.invoke.side_effect = ValueError("all retries exhausted")
        response = test_client.post("/analyze", json={"prompt": "trigger error"})
        assert response.status_code == 500
        assert response.json()["detail"] == "Internal processing error"
        assert "retries" not in response.text


class TestHealthEndpoint:
    def test_health_returns_200(self, client) -> None:
        test_client, _ = client
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_health_status_is_ok(self, client) -> None:
        test_client, _ = client
        response = test_client.get("/health")
        assert response.json()["status"] == "ok"

    def test_health_model_loaded_false_when_attribute_absent(self, client) -> None:
        test_client, _ = client
        if hasattr(test_client.app.state, "model_loaded"):
            del test_client.app.state.model_loaded
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is False


class TestDemoEndpoint:
    def test_root_serves_demo_html(self, client) -> None:
        test_client, _ = client
        response = test_client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "firewall" in response.text.lower()


class TestSchemaValidation:
    def test_prompt_min_length_rejects_empty(self) -> None:
        with pytest.raises(ValidationError, match="string_too_short"):
            AnalysisRequest(prompt="")

    def test_prompt_max_length_rejects_oversized(self) -> None:
        with pytest.raises(ValidationError, match="string_too_long"):
            AnalysisRequest(prompt="x" * 8193)

    def test_score_ge_rejects_negative(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal"):
            ClassificationScore(label="benign", score=-0.1)

    def test_score_le_rejects_above_one(self) -> None:
        with pytest.raises(ValidationError, match="less than or equal"):
            ClassificationScore(label="benign", score=1.1)

    def test_valid_score_accepted(self) -> None:
        s = ClassificationScore(label="injection", score=0.85)
        assert s.score == 0.85
