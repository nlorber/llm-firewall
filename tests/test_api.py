"""Integration tests for the FastAPI endpoints.

Covers:
- POST /analyze: valid request → 200 with correct AnalysisResponse schema
- POST /analyze: empty prompt → 422 validation error
- GET  /health: liveness probe → 200 with status "ok"
- Error propagation from the graph to the API response
"""
from __future__ import annotations

import pytest


class TestAnalyzeEndpoint:
    """Tests for ``POST /analyze``."""

    def test_valid_request_returns_200(self) -> None:
        raise NotImplementedError

    def test_empty_prompt_returns_422(self) -> None:
        raise NotImplementedError

    def test_response_body_matches_analysis_response_schema(self) -> None:
        raise NotImplementedError

    def test_pass_decision_present_in_response(self) -> None:
        raise NotImplementedError

    def test_block_decision_present_in_response(self) -> None:
        raise NotImplementedError


class TestHealthEndpoint:
    """Tests for ``GET /health``."""

    def test_health_returns_200(self) -> None:
        raise NotImplementedError

    def test_health_status_is_ok(self) -> None:
        raise NotImplementedError
