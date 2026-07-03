# src/firewall/judge/local_judge.py
"""A Judge backed by a local MLX model — mirrors LLMJudge's interface.

All MLX calls live behind ``_generate`` (lazy-imported), so this module imports on any
platform, unit tests can mock generation without MLX installed, and the real
Apple-Silicon path is exercised only by the integration smoke.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal

from firewall.judge.base import JudgeVerdict, build_judge_messages, parse_verdict

if TYPE_CHECKING:
    from firewall.judge.base import ChatMessage

_DEFAULT_MAX_TOKENS = 256
_THINK_OPEN = "<think>"
# A leading EMPTY think block — the Qwen3 chat template emits `<think>\n\n</think>` as
# structural scaffolding. Some fine-tuned checkpoints (e.g. Qwen3-4B-Instruct, whose
# template puts the block in the training target but not the generation prompt) reproduce
# it in their output; it is a benign artifact, not reasoning, so we strip rather than reject.
_EMPTY_THINK_RE = re.compile(r"^\s*<think>\s*</think>\s*", re.DOTALL)


class ThinkingModeError(RuntimeError):
    """Raised when a local model emits a NON-empty <think> block (real reasoning leakage).

    The judge requires a non-thinking model. Actual reasoning inside a think block buries
    the JSON behind an unbounded prefix, corrupting schema-validity and the decision-token
    signal, so we fail loudly. An *empty* ``<think></think>`` is a template artifact and is
    stripped instead (see :func:`strip_and_check_thinking`).
    """


def strip_and_check_thinking(raw: str) -> str:
    """Strip a leading empty ``<think></think>`` block; raise on a non-empty one.

    Returns the output with the benign empty block removed so the JSON parses; a think
    block containing any non-whitespace content is real reasoning leakage → ThinkingModeError.
    """
    stripped = _EMPTY_THINK_RE.sub("", raw, count=1)
    if _THINK_OPEN in stripped:
        raise ThinkingModeError(
            "local model emitted a non-empty <think> block; use a non-thinking model"
        )
    return stripped


class LocalJudge:
    """Judge a GRAY-zone prompt with a local MLX model (greedy, non-thinking)."""

    def __init__(
        self,
        model_path: str,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        enable_thinking: bool = False,
        on_failure: Literal["raise", "block"] = "raise",
        resample_temp: float | None = None,
        adapter_path: str | None = None,
    ) -> None:
        self._model_path = model_path
        self._max_tokens = max_tokens
        self._enable_thinking = enable_thinking
        self._on_failure = on_failure
        self._resample_temp = resample_temp
        self._adapter_path = adapter_path  # base + LoRA adapter (fine-tuned); None = base only
        self._model: Any = None
        self._tokenizer: Any = None

    def _load_kwargs(self) -> dict[str, Any]:
        """Kwargs for ``mlx_lm.load`` — forward ``adapter_path`` only when set so the base
        model still loads exactly as before (backward-compatible)."""
        kwargs: dict[str, Any] = {}
        if self._adapter_path is not None:
            kwargs["adapter_path"] = self._adapter_path
        return kwargs

    def judge(
        self,
        prompt: str,
        classification_label: str,
        scores: dict[str, float],
    ) -> JudgeVerdict:
        """Run the local model and parse a PASS/BLOCK verdict (see _parse_or_recover)."""
        messages, _boundary = build_judge_messages(prompt, classification_label, scores)
        raw = self._generate(messages, temp=0.0)
        return self._parse_or_recover(raw, messages)

    def generate_raw(
        self,
        prompt: str,
        classification_label: str,
        scores: dict[str, float],
    ) -> str:
        """Return the single greedy generation — no thinking check, no parse, no recovery.

        The eval harness needs the model's *actual* output to measure schema-validity
        before any repair; the no-`<think>` assertion and PASS/BLOCK parsing are policies
        that belong to the caller, so this stays a clean "give me the greedy output" seam.
        """
        messages, _boundary = build_judge_messages(prompt, classification_label, scores)
        return self._generate(messages, temp=0.0)

    def _parse_or_recover(self, raw: str, messages: list[ChatMessage]) -> JudgeVerdict:
        """Parse the greedy output; on schema failure optionally resample once, then
        apply the failure policy. A temp-0 retry would reproduce identical invalid output,
        so recovery (when enabled) resamples at ``resample_temp`` > 0.
        """
        try:
            return parse_verdict(strip_and_check_thinking(raw))
        except (ValueError, KeyError) as first_exc:
            if self._resample_temp is not None:
                retry_raw = self._generate(messages, temp=self._resample_temp)
                try:
                    return parse_verdict(strip_and_check_thinking(retry_raw))
                except (ValueError, KeyError):
                    pass
            if self._on_failure == "block":
                return JudgeVerdict(
                    decision="BLOCK",
                    reasoning="local judge produced no valid verdict; failing closed",
                    confidence=1.0,
                )
            raise ValueError(f"local judge produced invalid verdict: {raw!r}") from first_exc

    # The methods below are the only place MLX is touched; they are exercised by the
    # integration smoke (skipped on CI), so they are excluded from unit coverage.
    def _generate(self, messages: list[ChatMessage], temp: float) -> str:  # pragma: no cover
        """Render the chat template and generate with a greedy/temp sampler."""
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        self._ensure_loaded()
        text_prompt = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=self._enable_thinking,
        )
        sampler = make_sampler(temp=temp)
        result: str = generate(
            self._model,
            self._tokenizer,
            prompt=text_prompt,
            max_tokens=self._max_tokens,
            sampler=sampler,
        )
        return result.strip()

    def _ensure_loaded(self) -> None:  # pragma: no cover
        if self._model is None:
            from mlx_lm import load

            loaded = load(self._model_path, **self._load_kwargs())
            self._model, self._tokenizer = loaded[0], loaded[1]
