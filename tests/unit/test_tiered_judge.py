from __future__ import annotations

from firewall.judge.base import Judge, JudgeVerdict
from firewall.judge.tiered import (
    EscalationReason,
    LocalResult,
    Tier,
    TieredJudge,
)

_LOCAL_V = JudgeVerdict("PASS", "local says pass", 0.9)
_CLAUDE_V = JudgeVerdict("BLOCK", "claude says block", 0.8)


class _FakeLocal:
    """Returns a canned LocalResult, or raises if `boom` is set."""

    def __init__(self, result: LocalResult | None = None, boom: bool = False) -> None:
        self._result = result
        self._boom = boom
        self.calls = 0

    def judge_for_tiering(
        self, prompt: str, classification_label: str, scores: dict[str, float]
    ) -> LocalResult:
        self.calls += 1
        if self._boom:
            raise RuntimeError("local model exploded")
        assert self._result is not None
        return self._result


class _FakeClaude:
    def __init__(self) -> None:
        self.calls = 0

    def judge(
        self, prompt: str, classification_label: str, scores: dict[str, float]
    ) -> JudgeVerdict:
        self.calls += 1
        return _CLAUDE_V


def _tiered(local: _FakeLocal, claude: _FakeClaude, threshold: float = 0.5) -> TieredJudge:
    return TieredJudge(local=local, escalate_to=claude, threshold=threshold)


def test_keeps_local_when_confident() -> None:
    local = _FakeLocal(LocalResult(verdict=_LOCAL_V, signal=0.1, valid=True))
    claude = _FakeClaude()
    out = _tiered(local, claude).decide("p", "injection", {"injection": 0.5})
    assert out.tier is Tier.LOCAL
    assert out.reason is EscalationReason.NONE
    assert out.verdict is _LOCAL_V
    assert claude.calls == 0  # no escalation


def test_escalates_on_uncertainty() -> None:
    local = _FakeLocal(LocalResult(verdict=_LOCAL_V, signal=0.9, valid=True))
    claude = _FakeClaude()
    out = _tiered(local, claude).decide("p", "injection", {"injection": 0.5})
    assert out.tier is Tier.CLAUDE
    assert out.reason is EscalationReason.UNCERTAINTY
    assert out.verdict is _CLAUDE_V
    assert claude.calls == 1


def test_escalates_on_schema_invalid() -> None:
    local = _FakeLocal(LocalResult(verdict=None, signal=0.0, valid=False))
    claude = _FakeClaude()
    out = _tiered(local, claude).decide("p", "injection", {"injection": 0.5})
    assert out.reason is EscalationReason.SCHEMA_INVALID
    assert out.verdict is _CLAUDE_V


def test_escalates_on_local_error() -> None:
    local = _FakeLocal(boom=True)
    claude = _FakeClaude()
    out = _tiered(local, claude).decide("p", "injection", {"injection": 0.5})
    assert out.reason is EscalationReason.ERROR
    assert out.tier is Tier.CLAUDE
    assert out.verdict is _CLAUDE_V


def test_threshold_boundary_escalates() -> None:
    # signal == threshold escalates (>=), so the boundary is treated as uncertain.
    local = _FakeLocal(LocalResult(verdict=_LOCAL_V, signal=0.5, valid=True))
    out = _tiered(local, _FakeClaude(), threshold=0.5).decide("p", "x", {"x": 0.5})
    assert out.reason is EscalationReason.UNCERTAINTY


def test_judge_returns_decide_verdict_and_satisfies_protocol() -> None:
    local = _FakeLocal(LocalResult(verdict=_LOCAL_V, signal=0.1, valid=True))
    tiered = _tiered(local, _FakeClaude())
    assert isinstance(tiered, Judge)
    assert tiered.judge("p", "injection", {"injection": 0.5}) is _LOCAL_V
