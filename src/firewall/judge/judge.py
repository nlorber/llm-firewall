"""LLM-as-judge implementation using the Anthropic Claude API.

Called only for GRAY zone prompts where the classifier score is ambiguous.
Returns a structured JSON decision with reasoning.

Prompt contract:
    Input:  original prompt + classifier label + per-class confidence scores
    Output: JSON with keys ``decision`` ("PASS"|"BLOCK"), ``reasoning``, ``confidence``
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class JudgeVerdict:
    """Structured verdict returned by the LLM judge.

    Attributes:
        decision: ``"PASS"`` or ``"BLOCK"``.
        reasoning: Human-readable explanation of the decision.
        confidence: Judge's self-reported confidence in range [0, 1].
    """

    decision: str
    reasoning: str
    confidence: float


class LLMJudge:
    """Wraps the Anthropic Claude API with a structured prompt template.

    Args:
        model: Claude model identifier.
        max_tokens: Maximum tokens for the judge response.
        retry_count: Number of retries on API errors or malformed JSON.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 512,
        retry_count: int = 2,
    ) -> None:
        raise NotImplementedError

    def judge(
        self,
        prompt: str,
        classification_label: str,
        scores: dict[str, float],
    ) -> JudgeVerdict:
        """Ask Claude whether a GRAY zone prompt should be blocked.

        Args:
            prompt: The original user prompt under evaluation.
            classification_label: Top predicted label from the classifier.
            scores: Per-class probability dict from the classifier.

        Returns:
            :class:`JudgeVerdict` with decision, reasoning, and confidence.

        Raises:
            ValueError: If the API returns unparseable JSON after all retries.
        """
        raise NotImplementedError
