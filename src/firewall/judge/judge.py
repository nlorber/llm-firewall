# src/firewall/judge/judge.py
from __future__ import annotations

import json
import time
from typing import Any

import anthropic

from firewall.judge.base import (
    JudgeVerdict,
    build_judge_messages,
    parse_verdict,
)

# Re-exported so existing imports (`from firewall.judge.judge import JudgeVerdict`) keep
# working now that the dataclass lives in firewall.judge.base.
__all__ = ["JudgeVerdict", "LLMJudge"]

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_DEFAULT_MAX_TOKENS = 512
_DEFAULT_RETRY_COUNT = 2
_DEFAULT_TIMEOUT_SECONDS = 10.0
_DEFAULT_BACKOFF_BASE_SECONDS = 0.5
_DEFAULT_TEMPERATURE: float | None = None


class LLMJudge:
    """Invokes Claude to make a final PASS/BLOCK decision on GRAY zone prompts."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        retry_count: int = _DEFAULT_RETRY_COUNT,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        backoff_base: float = _DEFAULT_BACKOFF_BASE_SECONDS,
        temperature: float | None = _DEFAULT_TEMPERATURE,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._retry_count = retry_count
        self._timeout = timeout
        self._backoff_base = backoff_base
        self._temperature = temperature
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
        verdict, _raw, _usage = self.judge_verbose(prompt, classification_label, scores)
        return verdict

    def judge_verbose(
        self,
        prompt: str,
        classification_label: str,
        scores: dict[str, float],
    ) -> tuple[JudgeVerdict, str, dict[str, int]]:
        """Judge and also return the raw response text and token usage.

        Used by the distillation eval harness for the reference row's real cost/latency;
        ``judge`` is the thin wrapper. Same retry/parse semantics as before.
        """
        # Shared builder = single source of truth for what the judge sees. The
        # attacker-controlled prompt is sealed in a per-call nonce tag; the system
        # prompt tells the judge to treat the tagged content strictly as data.
        messages, _boundary = build_judge_messages(prompt, classification_label, scores)
        system_prompt = messages[0]["content"]
        user_message = messages[1]["content"]

        last_exc: Exception | None = None
        raw: str = ""
        for attempt in range(self._retry_count + 1):
            try:
                extra: dict[str, Any] = {}
                if self._temperature is not None:
                    extra["temperature"] = self._temperature
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                    timeout=self._timeout,
                    **extra,
                )
                block = response.content[0]
                raw = block.text.strip()  # first content block is always a TextBlock
                return parse_verdict(raw), raw, _usage_of(response)
            except (
                anthropic.APIError,
                json.JSONDecodeError,
                KeyError,
                ValueError,
                IndexError,
                AttributeError,
            ) as exc:
                last_exc = exc
                # Exponential backoff before the next attempt (none after the last).
                # Deterministic for predictable latency bounds; add jitter if these
                # judges ever fan out across many concurrent workers.
                if attempt < self._retry_count:
                    time.sleep(self._backoff_base * (2**attempt))

        raise ValueError(
            f"failed to obtain judge verdict after {self._retry_count + 1} attempts. "
            f"Last response: {raw!r}. Last error: {last_exc}"
        )


def _usage_of(response: Any) -> dict[str, int]:
    """Extract input/output token counts, defaulting to 0 when absent or non-integer.

    Read defensively so a bare ``MagicMock`` response (unit tests) yields zeros rather
    than crashing; real Anthropic responses carry integer ``usage.{input,output}_tokens``.
    """
    usage = getattr(response, "usage", None)
    in_tok = getattr(usage, "input_tokens", 0)
    out_tok = getattr(usage, "output_tokens", 0)
    return {
        "input_tokens": in_tok if isinstance(in_tok, int) else 0,
        "output_tokens": out_tok if isinstance(out_tok, int) else 0,
    }
