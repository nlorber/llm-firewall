# tests/test_judge.py
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import anthropic
import pytest

from firewall.judge.judge import JudgeVerdict, LLMJudge


class _MockAPIError(anthropic.APIError):
    """Constructable APIError for use as a mock side_effect."""

    def __init__(self, message: str = "mock api error") -> None:
        Exception.__init__(self, message)


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

    @pytest.fixture(autouse=True)
    def mock_sleep(self) -> Any:
        """Patch backoff sleep so retry tests don't actually wait."""
        with patch("firewall.judge.judge.time.sleep") as m:
            yield m

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

        with pytest.raises(ValueError, match="failed to obtain judge verdict"):
            judge.judge("x", "injection", {"injection": 0.5})

        assert judge._client.messages.create.call_count == 3

    def test_code_fence_wrapped_json_is_parsed(self, judge: LLMJudge) -> None:
        payload = '```json\n{"decision": "PASS", "reasoning": "safe", "confidence": 0.85}\n```'
        judge._client.messages.create.return_value = _mock_anthropic_response(payload)

        result = judge.judge("hello", "benign", {"benign": 0.45})
        assert result.decision == "PASS"
        assert result.confidence == pytest.approx(0.85)

    def test_confidence_outside_valid_range_is_passed_through(self, judge: LLMJudge) -> None:
        payload = json.dumps({"decision": "BLOCK", "reasoning": "test", "confidence": 1.5})
        judge._client.messages.create.return_value = _mock_anthropic_response(payload)
        result = judge.judge("test", "injection", {"injection": 0.5})
        assert result.confidence == pytest.approx(1.5)

    def test_api_error_triggers_retry(self, judge: LLMJudge) -> None:
        good_payload = json.dumps({"decision": "PASS", "reasoning": "ok", "confidence": 0.8})
        judge._client.messages.create.side_effect = [
            _MockAPIError("transient"),
            _mock_anthropic_response(good_payload),
        ]
        result = judge.judge("hello", "benign", {"benign": 0.45})
        assert result.decision == "PASS"
        assert judge._client.messages.create.call_count == 2

    def test_all_api_errors_raise_after_retries_exhausted(self, judge: LLMJudge) -> None:
        judge._client.messages.create.side_effect = _MockAPIError("down")

        with pytest.raises(ValueError, match="failed to obtain judge verdict"):
            judge.judge("x", "injection", {"injection": 0.5})

        assert judge._client.messages.create.call_count == 3

    def test_backoff_grows_between_retries(self, judge: LLMJudge, mock_sleep: Any) -> None:
        good = json.dumps({"decision": "PASS", "reasoning": "ok", "confidence": 0.8})
        judge._client.messages.create.side_effect = [
            _MockAPIError("transient"),
            _MockAPIError("transient"),
            _mock_anthropic_response(good),
        ]
        result = judge.judge("hi", "benign", {"benign": 0.45})

        assert result.decision == "PASS"
        # Two failed attempts → exponential backoff: 0.5 * 2^0, then 0.5 * 2^1.
        assert [c.args[0] for c in mock_sleep.call_args_list] == [0.5, 1.0]

    def test_no_backoff_when_first_attempt_succeeds(
        self, judge: LLMJudge, mock_sleep: Any
    ) -> None:
        payload = json.dumps({"decision": "PASS", "reasoning": "ok", "confidence": 0.8})
        judge._client.messages.create.return_value = _mock_anthropic_response(payload)

        judge.judge("hello", "benign", {"benign": 0.45})
        mock_sleep.assert_not_called()

    def test_timeout_is_passed_to_api_call(self) -> None:
        with patch("firewall.judge.judge.anthropic.Anthropic"):
            judge = LLMJudge(timeout=7.5)
        payload = json.dumps({"decision": "PASS", "reasoning": "ok", "confidence": 0.8})
        judge._client.messages.create.return_value = _mock_anthropic_response(payload)

        judge.judge("hello", "benign", {"benign": 0.45})

        kwargs = judge._client.messages.create.call_args.kwargs
        assert kwargs.get("timeout") == 7.5
