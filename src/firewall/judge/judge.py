# src/firewall/judge/judge.py
from __future__ import annotations

import json
import re
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


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", re.DOTALL)

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_DEFAULT_MAX_TOKENS = 512
_DEFAULT_RETRY_COUNT = 2
_DEFAULT_TIMEOUT_SECONDS = 10.0


class LLMJudge:
    """Invokes Claude to make a final PASS/BLOCK decision on GRAY zone prompts."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        retry_count: int = _DEFAULT_RETRY_COUNT,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._retry_count = retry_count
        self._timeout = timeout
        self._client = anthropic.Anthropic()

    def judge(
        self,
        prompt: str,
        classification_label: str,
        scores: dict[str, float],
    ) -> JudgeVerdict:
        """Ask Claude whether a GRAY zone prompt should be blocked.

        Raises:
            ValueError: If all retries fail (API or parse errors).
        """
        scores_str = ", ".join(f"{k}={v:.2f}" for k, v in sorted(scores.items()))
        user_message = (
            f"Prompt: {prompt!r}\n"
            f"Classifier prediction: {classification_label}\n"
            f"Confidence scores: {scores_str}"
        )

        last_exc: Exception | None = None
        raw: str = ""
        for _attempt in range(self._retry_count + 1):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}],
                    timeout=self._timeout,
                )
                block = response.content[0]
                raw = block.text.strip()  # type: ignore[union-attr]  # we only send text prompts; first block is always TextBlock
                fence_match = _CODE_FENCE_RE.match(raw)
                cleaned = fence_match.group(1).strip() if fence_match else raw
                data = json.loads(cleaned)
                return JudgeVerdict(
                    decision=data["decision"],
                    reasoning=data["reasoning"],
                    confidence=float(data["confidence"]),
                )
            except (
                anthropic.APIError,
                json.JSONDecodeError,
                KeyError,
                ValueError,
                IndexError,
                AttributeError,
            ) as exc:
                last_exc = exc

        raise ValueError(
            f"failed to obtain judge verdict after {self._retry_count + 1} attempts. "
            f"Last response: {raw!r}. Last error: {last_exc}"
        )
