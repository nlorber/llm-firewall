# tests/unit/judge/test_judge_fallback.py
"""Unit tests for LLMJudge JSON-parse fallback and hallucination guard."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from firewall.judge.judge import LLMJudge


def _mock_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=content)]
    return msg


@pytest.fixture()
def judge() -> LLMJudge:
    with patch("firewall.judge.judge.anthropic.Anthropic"):
        return LLMJudge(model="claude-haiku-4-5-20251001", max_tokens=128, retry_count=2)


class TestCodeFenceRegex:
    """Cover _CODE_FENCE_RE (judge.py line 26) exhaustively."""

    def test_plain_json_no_fence(self, judge: LLMJudge) -> None:
        payload = json.dumps({"decision": "PASS", "reasoning": "ok", "confidence": 0.8})
        judge._client.messages.create.return_value = _mock_response(payload)
        result = judge.judge("hi", "benign", {"benign": 0.5})
        assert result.decision == "PASS"

    def test_code_fence_json_tag(self, judge: LLMJudge) -> None:
        payload = '```json\n{"decision": "BLOCK", "reasoning": "bad", "confidence": 0.9}\n```'
        judge._client.messages.create.return_value = _mock_response(payload)
        result = judge.judge("attack", "injection", {"injection": 0.6})
        assert result.decision == "BLOCK"

    def test_code_fence_no_tag(self, judge: LLMJudge) -> None:
        payload = '```\n{"decision": "PASS", "reasoning": "safe", "confidence": 0.7}\n```'
        judge._client.messages.create.return_value = _mock_response(payload)
        result = judge.judge("hello", "benign", {"benign": 0.5})
        assert result.decision == "PASS"


class TestMixedProseAndJSON:
    """Cover the scenario where the model prefixes prose before the JSON."""

    def test_prose_prefix_fails_and_retries(self, judge: LLMJudge) -> None:
        """Prose+JSON is not valid JSON and not a code fence — triggers a retry."""
        good = json.dumps({"decision": "PASS", "reasoning": "ok", "confidence": 0.8})
        judge._client.messages.create.side_effect = [
            _mock_response('Here is my verdict: {"decision": "PASS", "reasoning": "ok", "confidence": 0.8}'),
            _mock_response(good),
        ]
        result = judge.judge("test", "benign", {"benign": 0.5})
        assert result.decision == "PASS"
        assert judge._client.messages.create.call_count == 2

    def test_prose_only_exhausts_retries(self, judge: LLMJudge) -> None:
        """All responses are unparseable — raises ValueError after all attempts."""
        judge._client.messages.create.return_value = _mock_response(
            "I think this prompt is safe and should be passed."
        )
        with pytest.raises(ValueError, match="failed to obtain judge verdict"):
            judge.judge("test", "benign", {"benign": 0.5})
        assert judge._client.messages.create.call_count == 3


class TestRetryExhaustion:
    """Cover the 3-attempt retry loop exhaustion (retry_count=2 => 3 total calls)."""

    def test_three_attempts_on_bad_json(self, judge: LLMJudge) -> None:
        judge._client.messages.create.return_value = _mock_response("NOT_JSON")
        with pytest.raises(ValueError, match="failed to obtain judge verdict"):
            judge.judge("x", "injection", {"injection": 0.5})
        assert judge._client.messages.create.call_count == 3

    def test_error_message_contains_last_response(self, judge: LLMJudge) -> None:
        judge._client.messages.create.return_value = _mock_response("GARBAGE")
        with pytest.raises(ValueError, match="GARBAGE"):
            judge.judge("x", "injection", {"injection": 0.5})


class TestHallucinationGuard:
    """Cover the PASS/BLOCK validation added in item 4."""

    def test_invalid_decision_triggers_retry(self, judge: LLMJudge) -> None:
        good = json.dumps({"decision": "PASS", "reasoning": "ok", "confidence": 0.8})
        judge._client.messages.create.side_effect = [
            _mock_response(json.dumps({"decision": "MAYBE", "reasoning": "unsure", "confidence": 0.5})),
            _mock_response(good),
        ]
        result = judge.judge("test", "benign", {"benign": 0.5})
        assert result.decision == "PASS"
        assert judge._client.messages.create.call_count == 2

    def test_all_invalid_decisions_exhaust_retries(self, judge: LLMJudge) -> None:
        judge._client.messages.create.return_value = _mock_response(
            json.dumps({"decision": "UNKNOWN", "reasoning": "?", "confidence": 0.5})
        )
        with pytest.raises(ValueError, match="failed to obtain judge verdict"):
            judge.judge("x", "benign", {"benign": 0.5})
        assert judge._client.messages.create.call_count == 3
