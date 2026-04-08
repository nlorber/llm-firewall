# tests/test_judge.py
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from firewall.judge.judge import JudgeVerdict, LLMJudge


def _mock_anthropic_response(content: str) -> MagicMock:
    """Build a fake anthropic.messages.create() return value."""
    msg = MagicMock()
    msg.content = [MagicMock(text=content)]
    return msg


class TestLLMJudge:
    @pytest.fixture()
    def judge(self) -> LLMJudge:
        with patch("firewall.judge.judge.anthropic.Anthropic"):
            return LLMJudge(model="claude-haiku-4-5-20251001", max_tokens=128, retry_count=2)

    def test_judge_returns_judge_verdict(self, judge: LLMJudge) -> None:
        payload = json.dumps({"decision": "BLOCK", "reasoning": "injection attempt", "confidence": 0.95})
        judge._client.messages.create.return_value = _mock_anthropic_response(payload)

        result = judge.judge("ignore prev", "injection", {"benign": 0.1, "injection": 0.55})
        assert isinstance(result, JudgeVerdict)

    def test_pass_decision_propagated(self, judge: LLMJudge) -> None:
        payload = json.dumps({"decision": "PASS", "reasoning": "benign", "confidence": 0.9})
        judge._client.messages.create.return_value = _mock_anthropic_response(payload)

        result = judge.judge("hello world", "benign", {"benign": 0.45, "injection": 0.35})
        assert result.decision == "PASS"
        assert result.confidence == pytest.approx(0.9)

    def test_block_decision_propagated(self, judge: LLMJudge) -> None:
        payload = json.dumps({"decision": "BLOCK", "reasoning": "dangerous", "confidence": 0.98})
        judge._client.messages.create.return_value = _mock_anthropic_response(payload)

        result = judge.judge("jailbreak prompt", "jailbreak", {"jailbreak": 0.6, "benign": 0.2})
        assert result.decision == "BLOCK"

    def test_malformed_json_triggers_retry(self, judge: LLMJudge) -> None:
        good_payload = json.dumps({"decision": "PASS", "reasoning": "ok", "confidence": 0.8})
        judge._client.messages.create.side_effect = [
            _mock_anthropic_response("NOT_JSON"),        # first call fails
            _mock_anthropic_response(good_payload),      # second call succeeds
        ]
        result = judge.judge("hello", "benign", {"benign": 0.45})
        assert result.decision == "PASS"
        assert judge._client.messages.create.call_count == 2

    def test_raises_value_error_after_all_retries_exhausted(self, judge: LLMJudge) -> None:
        judge._client.messages.create.return_value = _mock_anthropic_response("NOT_JSON")

        with pytest.raises(ValueError, match="failed to parse judge response"):
            judge.judge("x", "injection", {"injection": 0.5})

        assert judge._client.messages.create.call_count == 3
