"""Unit tests for the LLM judge with a mocked Claude API.

Covers:
- Correct structured output parsing into JudgeVerdict
- PASS / BLOCK decision propagation
- Retry logic on malformed JSON
- ValueError raised after all retries exhausted
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


class TestLLMJudge:
    """Tests for :class:`firewall.judge.judge.LLMJudge`."""

    def test_judge_returns_judge_verdict_dataclass(self) -> None:
        raise NotImplementedError

    def test_pass_decision_propagated_correctly(self) -> None:
        raise NotImplementedError

    def test_block_decision_propagated_correctly(self) -> None:
        raise NotImplementedError

    def test_malformed_json_triggers_retry(self) -> None:
        raise NotImplementedError

    def test_raises_value_error_after_all_retries_exhausted(self) -> None:
        raise NotImplementedError
