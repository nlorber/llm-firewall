# src/firewall/judge/tiered.py
"""TieredJudge — run the local judge first, escalate only the uncertain verdicts to Claude.

The composite implements the same ``Judge`` protocol as its parts, so the orchestrator holds
it without knowing there are two tiers. It is dependency-light (no MLX, no Anthropic import):
the local tier is injected as a ``TieringLocalJudge`` and the escalation target as any
``Judge``, so the whole escalation policy is unit-tested with fakes.

Escalation happens when the local model is uncertain (``signal >= threshold``), emits
un-parseable output (schema-invalid), or fails (error). The signal convention is
higher = more uncertain. Tier + reason are metadata for metrics/logging; the returned
verdict keeps the plain ``{decision, reasoning, confidence}`` schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from firewall.judge.base import Judge, JudgeVerdict

_DEFAULT_TEACHER = "claude-haiku-4-5-20251001"  # mirrors LLMJudge's default
_VALID_SIGNAL_MODES = ("confidence", "logprob_margin", "entropy")  # LocalJudge signal_mode values


class Tier(StrEnum):
    """Which tier produced the returned verdict."""

    LOCAL = "local"
    CLAUDE = "claude"


class EscalationReason(StrEnum):
    """Why (or why not) the verdict was escalated — the decomposition the eval reports."""

    NONE = "none"  # kept local
    UNCERTAINTY = "uncertainty"  # signal past threshold (the only "genuine" leakage)
    SCHEMA_INVALID = "schema_invalid"  # local output did not parse
    ERROR = "error"  # local model raised


@dataclass
class LocalResult:
    """What the local tier returns for tiering: verdict (None if unparseable), an uncertainty
    signal (higher = more uncertain), and whether the output parsed."""

    verdict: JudgeVerdict | None
    signal: float
    valid: bool


@runtime_checkable
class TieringLocalJudge(Protocol):
    """A local judge that also reports an uncertainty signal for escalation."""

    def judge_for_tiering(
        self, prompt: str, classification_label: str, scores: dict[str, float]
    ) -> LocalResult: ...


@dataclass
class TieredOutcome:
    """The verdict plus the tier/reason/signal metadata the eval and logs record."""

    verdict: JudgeVerdict
    tier: Tier
    reason: EscalationReason
    signal: float


class TieredJudge:
    """Local-first judge that escalates uncertain/invalid/failed verdicts to ``escalate_to``."""

    def __init__(
        self,
        local: TieringLocalJudge,
        escalate_to: Judge,
        threshold: float,
    ) -> None:
        self._local = local
        self._escalate_to = escalate_to
        self._threshold = threshold

    def judge(
        self, prompt: str, classification_label: str, scores: dict[str, float]
    ) -> JudgeVerdict:
        """The ``Judge`` interface — just the verdict; see ``decide`` for the metadata."""
        return self.decide(prompt, classification_label, scores).verdict

    def decide(
        self, prompt: str, classification_label: str, scores: dict[str, float]
    ) -> TieredOutcome:
        """Run the local tier, then escalate on error / schema-invalid / uncertainty."""
        try:
            result = self._local.judge_for_tiering(prompt, classification_label, scores)
        except Exception:  # noqa: BLE001 — any local failure escalates rather than 500s
            return self._escalate(prompt, classification_label, scores, EscalationReason.ERROR)

        if not result.valid or result.verdict is None:
            return self._escalate(
                prompt,
                classification_label,
                scores,
                EscalationReason.SCHEMA_INVALID,
                result.signal,
            )
        if result.signal >= self._threshold:
            return self._escalate(
                prompt, classification_label, scores, EscalationReason.UNCERTAINTY, result.signal
            )
        return TieredOutcome(result.verdict, Tier.LOCAL, EscalationReason.NONE, result.signal)

    def _escalate(
        self,
        prompt: str,
        classification_label: str,
        scores: dict[str, float],
        reason: EscalationReason,
        signal: float = float("nan"),
    ) -> TieredOutcome:
        verdict = self._escalate_to.judge(prompt, classification_label, scores)
        return TieredOutcome(verdict, Tier.CLAUDE, reason, signal)


def make_judge(
    backend: str,
    *,
    teacher_model: str = _DEFAULT_TEACHER,
    temperature: float | None = None,
    teacher_max_tokens: int = 512,
    retry_count: int = 2,
    timeout: float = 10.0,
    local_model: str | None = None,
    adapter_path: str | None = None,
    signal_mode: Literal["confidence", "logprob_margin", "entropy"] = "logprob_margin",
    threshold: float = 0.5,
    max_tokens: int = 256,
) -> Judge:
    """Build the judge for a configured backend (``claude`` | ``local`` | ``tiered``).

    ``local``/``tiered`` require ``local_model``; ``tiered`` wraps a signal-emitting LocalJudge
    plus a Claude escalation target. Judges are constructed lazily (no MLX/Anthropic import at
    module load), so a missing adapter or MLX surfaces at first use — the orchestrator (Plan 7)
    turns that into a startup check. ``local``-only fails closed to BLOCK on unrecoverable error.
    """
    from firewall.judge.judge import LLMJudge

    def _claude() -> LLMJudge:
        return LLMJudge(
            model=teacher_model,
            max_tokens=teacher_max_tokens,
            retry_count=retry_count,
            timeout=timeout,
            temperature=temperature,
        )

    if backend == "claude":
        return _claude()
    if backend in ("local", "tiered"):
        from firewall.judge.local_judge import LocalJudge

        if not local_model:
            raise ValueError(f"judge backend {backend!r} requires local_model")
        if backend == "local":
            return LocalJudge(
                local_model, adapter_path=adapter_path, max_tokens=max_tokens, on_failure="block"
            )
        # Validate here rather than let a typo'd config value silently fall through to the
        # margin path (any non-"entropy" value hits it) and misreport the escalation signal.
        if signal_mode not in _VALID_SIGNAL_MODES:
            raise ValueError(
                f"invalid escalation signal {signal_mode!r}; expected one of {_VALID_SIGNAL_MODES}"
            )
        local = LocalJudge(
            local_model,
            adapter_path=adapter_path,
            signal_mode=signal_mode,
            max_tokens=max_tokens,
        )
        return TieredJudge(local, _claude(), threshold)
    raise ValueError(f"unknown judge backend: {backend!r}")
