# src/firewall/judge/judge.py
from __future__ import annotations

import json
from dataclasses import dataclass

import anthropic

_SYSTEM_PROMPT = """\
You are a security classifier judge. You receive a user prompt, its classifier prediction, \
and per-class confidence scores. Decide whether the prompt should be PASSED to an LLM or BLOCKED.

Respond ONLY with a JSON object — no prose, no markdown:
{"decision": "PASS" or "BLOCK", "reasoning": "<1-2 sentences>", "confidence": <0.0-1.0>}"""


@dataclass
class JudgeVerdict:
    """Structured verdict from the LLM judge."""
    decision: str      # "PASS" | "BLOCK"
    reasoning: str
    confidence: float


class LLMJudge:
    """Invokes Claude to make a final PASS/BLOCK decision on GRAY zone prompts."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 512,
        retry_count: int = 2,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._retry_count = retry_count
        self._client = anthropic.Anthropic()

    def judge(
        self,
        prompt: str,
        classification_label: str,
        scores: dict[str, float],
    ) -> JudgeVerdict:
        """Ask Claude whether a GRAY zone prompt should be blocked.

        Raises:
            ValueError: If all retries produce unparseable JSON.
        """
        scores_str = ", ".join(f"{k}={v:.2f}" for k, v in sorted(scores.items()))
        user_message = (
            f"Prompt: {prompt!r}\n"
            f"Classifier prediction: {classification_label}\n"
            f"Confidence scores: {scores_str}"
        )

        last_exc: Exception | None = None
        for _attempt in range(self._retry_count + 1):
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            raw = response.content[0].text.strip()
            try:
                data = json.loads(raw)
                return JudgeVerdict(
                    decision=data["decision"],
                    reasoning=data["reasoning"],
                    confidence=float(data["confidence"]),
                )
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                last_exc = exc

        raise ValueError(
            f"failed to parse judge response after {self._retry_count + 1} attempts. "
            f"Last response: {raw!r}. Last error: {last_exc}"
        )
