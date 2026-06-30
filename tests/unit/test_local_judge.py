from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from firewall.judge.base import Judge, JudgeVerdict
from firewall.judge.local_judge import LocalJudge

if TYPE_CHECKING:
    from collections.abc import Callable

    from firewall.judge.base import ChatMessage


def _fixed(text: str) -> Callable[[list[ChatMessage], float], str]:
    """A fake _generate that ignores inputs and returns a canned model output."""

    def _gen(messages: list[ChatMessage], temp: float) -> str:
        return text

    return _gen


class TestLocalJudgeHappyPath:
    def test_parses_valid_block_verdict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        judge = LocalJudge("fake-model")
        payload = json.dumps({"decision": "BLOCK", "reasoning": "injection", "confidence": 0.9})
        monkeypatch.setattr(judge, "_generate", _fixed(payload))
        verdict = judge.judge("ignore prev", "injection", {"injection": 0.55, "benign": 0.2})
        assert isinstance(verdict, JudgeVerdict)
        assert verdict.decision == "BLOCK"
        assert verdict.confidence == pytest.approx(0.9)

    def test_pass_decision_propagated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        judge = LocalJudge("fake-model")
        payload = json.dumps({"decision": "PASS", "reasoning": "benign", "confidence": 0.8})
        monkeypatch.setattr(judge, "_generate", _fixed(payload))
        assert judge.judge("hello", "benign", {"benign": 0.45}).decision == "PASS"

    def test_conforms_to_judge_protocol(self) -> None:
        assert isinstance(LocalJudge("fake-model"), Judge)
