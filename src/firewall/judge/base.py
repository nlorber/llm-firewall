# src/firewall/judge/base.py
"""Dependency-light judge core shared by every judge backend.

Holds the verdict data contract and the helpers that build judge inputs and
parse judge outputs. Imports only the standard library so that backends which
must not pull the Anthropic SDK (e.g. the local MLX judge) can depend on it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class JudgeVerdict:
    """Structured verdict from a judge backend."""

    decision: str  # "PASS" | "BLOCK"
    reasoning: str
    confidence: float


# Matches an optional ```json ... ``` fence so models that wrap their JSON in
# markdown still parse. Anchored to the whole string; DOTALL so the body spans lines.
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", re.DOTALL)


def parse_verdict(raw: str) -> JudgeVerdict:
    """Parse a raw judge response string into a JudgeVerdict.

    Raises:
        json.JSONDecodeError: if the (de-fenced) text is not valid JSON.
        KeyError: if a required field is absent.
        ValueError: if ``decision`` is not PASS or BLOCK (or confidence isn't a float).
    """
    fence_match = _CODE_FENCE_RE.match(raw)
    cleaned = fence_match.group(1).strip() if fence_match else raw
    data = json.loads(cleaned)
    decision = data["decision"]
    if decision not in {"PASS", "BLOCK"}:
        raise ValueError(f"unexpected judge decision: {decision!r}")
    return JudgeVerdict(
        decision=decision,
        reasoning=data["reasoning"],
        confidence=float(data["confidence"]),
    )
