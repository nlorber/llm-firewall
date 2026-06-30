from __future__ import annotations

import json

import pytest

from firewall.judge.base import JudgeVerdict, parse_verdict


class TestParseVerdict:
    def test_parses_plain_json(self) -> None:
        raw = json.dumps({"decision": "BLOCK", "reasoning": "injection", "confidence": 0.9})
        verdict = parse_verdict(raw)
        assert isinstance(verdict, JudgeVerdict)
        assert verdict.decision == "BLOCK"
        assert verdict.reasoning == "injection"
        assert verdict.confidence == pytest.approx(0.9)

    def test_strips_json_code_fence(self) -> None:
        raw = '```json\n{"decision": "PASS", "reasoning": "safe", "confidence": 0.85}\n```'
        verdict = parse_verdict(raw)
        assert verdict.decision == "PASS"
        assert verdict.confidence == pytest.approx(0.85)

    def test_rejects_unexpected_decision(self) -> None:
        raw = json.dumps({"decision": "MAYBE", "reasoning": "x", "confidence": 0.5})
        with pytest.raises(ValueError, match="unexpected judge decision"):
            parse_verdict(raw)

    def test_raises_on_non_json(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            parse_verdict("NOT_JSON")

    def test_confidence_outside_range_passed_through(self) -> None:
        raw = json.dumps({"decision": "BLOCK", "reasoning": "t", "confidence": 1.5})
        assert parse_verdict(raw).confidence == pytest.approx(1.5)
