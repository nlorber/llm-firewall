from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from firewall.judge.base import Judge, JudgeVerdict
from firewall.judge.local_judge import LocalJudge, ThinkingModeError

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


class TestLocalJudgeGenerateRaw:
    def test_generate_raw_returns_output_verbatim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # No parse, no thinking-check, no recovery — even a leaked think block or garbage
        # comes straight back so the eval layer can score schema-validity itself.
        judge = LocalJudge("fake-model")
        monkeypatch.setattr(judge, "_generate", _fixed("<think>hmm</think> not json"))
        assert (
            judge.generate_raw("x", "injection", {"injection": 0.5})
            == "<think>hmm</think> not json"
        )


class TestLocalJudgeNonThinking:
    def test_rejects_think_block(self, monkeypatch: pytest.MonkeyPatch) -> None:
        judge = LocalJudge("fake-model")
        leaked = (
            '<think>let me reason</think>\n{"decision":"PASS","reasoning":"x","confidence":0.7}'
        )
        monkeypatch.setattr(judge, "_generate", _fixed(leaked))
        with pytest.raises(ThinkingModeError, match="think"):
            judge.judge("hello", "benign", {"benign": 0.45})


class TestLocalJudgeFailureModes:
    def test_invalid_raises_when_on_failure_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        judge = LocalJudge("fake-model", on_failure="raise")
        monkeypatch.setattr(judge, "_generate", _fixed("NOT JSON"))
        with pytest.raises(ValueError, match="invalid verdict"):
            judge.judge("x", "injection", {"injection": 0.5})

    def test_invalid_blocks_when_on_failure_block(self, monkeypatch: pytest.MonkeyPatch) -> None:
        judge = LocalJudge("fake-model", on_failure="block")
        monkeypatch.setattr(judge, "_generate", _fixed("NOT JSON"))
        verdict = judge.judge("x", "injection", {"injection": 0.5})
        assert verdict.decision == "BLOCK"
        assert verdict.confidence == pytest.approx(1.0)

    def test_resample_recovers_then_parses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        judge = LocalJudge("fake-model", on_failure="block", resample_temp=0.7)
        calls: dict[str, int] = {"n": 0}
        good = json.dumps({"decision": "PASS", "reasoning": "ok", "confidence": 0.8})

        def _gen(messages: list[ChatMessage], temp: float) -> str:
            calls["n"] += 1
            return "NOT JSON" if calls["n"] == 1 else good

        monkeypatch.setattr(judge, "_generate", _gen)
        verdict = judge.judge("hi", "benign", {"benign": 0.45})
        assert verdict.decision == "PASS"
        assert calls["n"] == 2  # greedy attempt failed, one resample succeeded

    def test_resample_failure_falls_through_to_policy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Both the greedy attempt and the resample produce invalid JSON → the failure
        # policy applies (here: fail-closed to BLOCK).
        judge = LocalJudge("fake-model", on_failure="block", resample_temp=0.7)
        monkeypatch.setattr(judge, "_generate", _fixed("STILL NOT JSON"))
        verdict = judge.judge("x", "injection", {"injection": 0.5})
        assert verdict.decision == "BLOCK"
