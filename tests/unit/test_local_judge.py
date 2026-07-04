from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from firewall.judge.base import Judge, JudgeVerdict
from firewall.judge.local_judge import (
    LocalJudge,
    ThinkingModeError,
    _decision_uncertainty,
    strip_and_check_thinking,
)

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


class TestLocalJudgeAdapter:
    def test_load_kwargs_omits_adapter_when_none(self) -> None:
        assert LocalJudge("base")._load_kwargs() == {}

    def test_load_kwargs_forwards_adapter_when_set(self) -> None:
        judge = LocalJudge("base", adapter_path="adapters/qwen3-1.7b")
        assert judge._load_kwargs() == {"adapter_path": "adapters/qwen3-1.7b"}


class TestLocalJudgeTiering:
    def test_confidence_signal_is_one_minus_confidence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        judge = LocalJudge("fake-model")
        payload = json.dumps({"decision": "BLOCK", "reasoning": "x", "confidence": 0.7})
        monkeypatch.setattr(judge, "_generate", _fixed(payload))
        result = judge.judge_for_tiering("x", "injection", {"injection": 0.5})
        assert result.valid
        assert result.verdict is not None and result.verdict.decision == "BLOCK"
        assert result.signal == pytest.approx(0.3)

    def test_invalid_output_is_max_uncertain_and_invalid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        judge = LocalJudge("fake-model")
        monkeypatch.setattr(judge, "_generate", _fixed("NOT JSON"))
        result = judge.judge_for_tiering("x", "injection", {"injection": 0.5})
        assert not result.valid
        assert result.verdict is None
        assert result.signal == 1.0

    def test_satisfies_tiering_local_judge_protocol(self) -> None:
        from firewall.judge.tiered import TieringLocalJudge

        assert isinstance(LocalJudge("fake-model"), TieringLocalJudge)


class TestDecisionUncertainty:
    def test_coin_flip_is_max_uncertainty(self) -> None:
        assert _decision_uncertainty(0.0, 0.0, "margin") == pytest.approx(1.0)
        assert _decision_uncertainty(0.0, 0.0, "entropy") == pytest.approx(1.0)

    def test_confident_is_near_zero(self) -> None:
        assert _decision_uncertainty(0.0, -20.0, "margin") == pytest.approx(0.0, abs=1e-6)
        assert _decision_uncertainty(0.0, -20.0, "entropy") == pytest.approx(0.0, abs=1e-6)

    def test_monotonic_in_margin(self) -> None:
        # A wider log-prob gap = more certain = lower uncertainty.
        assert _decision_uncertainty(0.0, -1.0, "margin") > _decision_uncertainty(
            0.0, -3.0, "margin"
        )
        assert _decision_uncertainty(-0.5, 0.0, "entropy") > _decision_uncertainty(
            -4.0, 0.0, "entropy"
        )


class TestLocalJudgeNonThinking:
    def test_rejects_non_empty_think_block(self, monkeypatch: pytest.MonkeyPatch) -> None:
        judge = LocalJudge("fake-model")
        leaked = (
            '<think>let me reason</think>\n{"decision":"PASS","reasoning":"x","confidence":0.7}'
        )
        monkeypatch.setattr(judge, "_generate", _fixed(leaked))
        with pytest.raises(ThinkingModeError, match="think"):
            judge.judge("hello", "benign", {"benign": 0.45})

    def test_tolerates_empty_think_block(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Qwen3-4B-Instruct fine-tunes reproduce the template's empty <think></think> in the
        # output; it is scaffolding, not reasoning, so it is stripped and the JSON still parses.
        judge = LocalJudge("fake-model")
        payload = (
            '<think>\n\n</think>\n\n{"decision": "BLOCK", "reasoning": "x", "confidence": 0.8}'
        )
        monkeypatch.setattr(judge, "_generate", _fixed(payload))
        assert judge.judge("x", "injection", {"injection": 0.5}).decision == "BLOCK"

    def test_strip_and_check_thinking_helper(self) -> None:
        assert strip_and_check_thinking('<think>\n\n</think>\n\n{"x": 1}') == '{"x": 1}'
        assert strip_and_check_thinking('{"x": 1}') == '{"x": 1}'
        with pytest.raises(ThinkingModeError):
            strip_and_check_thinking("<think>real reasoning</think>{}")


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
