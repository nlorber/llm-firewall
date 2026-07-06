# src/firewall/judge/local_judge.py
"""A Judge backed by a local MLX model — mirrors LLMJudge's interface.

All MLX calls live behind ``_generate`` (lazy-imported), so this module imports on any
platform, unit tests can mock generation without MLX installed, and the real
Apple-Silicon path is exercised only by the integration smoke.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any, Literal

from firewall.judge.base import JudgeVerdict, build_judge_messages, parse_verdict
from firewall.judge.tiered import LocalResult

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


def _decision_uncertainty(lp_block: float, lp_pass: float, mode: str) -> float:
    """Map the BLOCK/PASS decision-token log-probs to an uncertainty in [0, 1] (1 = a coin
    flip between the two, 0 = certain). ``entropy`` = binary entropy of the renormalised
    2-way distribution; otherwise a margin-based ``2 * min(p)``. Higher = more uncertain, so
    it composes with the ``signal >= threshold`` escalation rule.
    """
    top = max(lp_block, lp_pass)
    e_block = math.exp(lp_block - top)
    e_pass = math.exp(lp_pass - top)
    p_block = e_block / (e_block + e_pass)
    if mode == "entropy":
        if p_block <= 0.0 or p_block >= 1.0:
            return 0.0
        return -(p_block * math.log2(p_block) + (1 - p_block) * math.log2(1 - p_block))
    return 2.0 * min(p_block, 1.0 - p_block)


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
        signal_mode: Literal["confidence", "logprob_margin", "entropy"] = "confidence",
    ) -> None:
        self._model_path = model_path
        self._max_tokens = max_tokens
        self._enable_thinking = enable_thinking
        self._on_failure = on_failure
        self._resample_temp = resample_temp
        self._adapter_path = adapter_path  # base + LoRA adapter (fine-tuned); None = base only
        self._signal_mode = signal_mode  # tiering uncertainty signal (see judge_for_tiering)
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

    def judge_for_tiering(
        self,
        prompt: str,
        classification_label: str,
        scores: dict[str, float],
    ) -> LocalResult:
        """Judge and report an uncertainty signal (higher = more uncertain) for TieredJudge.

        In ``"confidence"`` mode the signal is ``1 - emitted confidence`` (a structurally weak
        proxy — the student mimics the teacher's confidence, not its own uncertainty). The
        ``logprob_margin`` / ``entropy`` modes instead read the model's *own* uncertainty at the
        BLOCK/PASS decision token. An unparseable output is maximally uncertain and invalid →
        the composite escalates.
        """
        messages, _boundary = build_judge_messages(prompt, classification_label, scores)
        logprob_signal: float | None = None
        if self._signal_mode == "confidence":
            raw = self._generate(messages, temp=0.0)
        else:
            raw, logprob_signal = self._generate_with_signal(messages)
        try:
            verdict = parse_verdict(strip_and_check_thinking(raw))
        except (ValueError, ThinkingModeError):
            return LocalResult(verdict=None, signal=1.0, valid=False)
        signal = (1.0 - verdict.confidence) if logprob_signal is None else logprob_signal
        return LocalResult(verdict=verdict, signal=signal, valid=True)

    def _parse_or_recover(self, raw: str, messages: list[ChatMessage]) -> JudgeVerdict:
        """Parse the greedy output; on failure optionally resample once, then apply the
        failure policy. Covers both schema errors and thinking-mode leakage: with
        ``on_failure="block"`` either mode fails closed to BLOCK, so an attacker-baited
        ``<think>`` block cannot escape the policy as an uncaught exception. A temp-0 retry
        would reproduce identical invalid output, so recovery (when enabled) resamples at
        ``resample_temp`` > 0.
        """
        try:
            return parse_verdict(strip_and_check_thinking(raw))
        except (ValueError, ThinkingModeError) as first_exc:
            if self._resample_temp is not None:
                retry_raw = self._generate(messages, temp=self._resample_temp)
                try:
                    return parse_verdict(strip_and_check_thinking(retry_raw))
                except (ValueError, ThinkingModeError):
                    pass
            if self._on_failure == "block":
                return JudgeVerdict(
                    decision="BLOCK",
                    reasoning="local judge produced no valid verdict; failing closed",
                    confidence=1.0,
                )
            if isinstance(first_exc, ThinkingModeError):
                raise  # surface reasoning leakage distinctly when not failing closed
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

    def _generate_with_signal(  # pragma: no cover
        self, messages: list[ChatMessage]
    ) -> tuple[str, float]:
        """Greedy-decode and read the model's uncertainty at the BLOCK/PASS decision token.

        Streams tokens; at the step that first emits a char past ``"decision": "`` (the JSON
        decision value), the yielded log-probs are the decision distribution — we read the
        BLOCK and PASS first-token log-probs there and map them to an uncertainty in [0, 1].
        Falls back to maximally uncertain if the model never emits a well-formed decision key.
        """
        import mlx.core as mx
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler

        self._ensure_loaded()
        tok = self._tokenizer
        prompt_ids = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=self._enable_thinking,
            tokenize=True,
        )
        block_first = tok.encode("BLOCK", add_special_tokens=False)[0]
        pass_first = tok.encode("PASS", add_special_tokens=False)[0]
        mode = "entropy" if self._signal_mode == "entropy" else "margin"
        marker = '"decision": "'

        tokens: list[int] = []
        signal = 1.0  # maximally uncertain until a decision token is located
        found = False
        sampler = make_sampler(temp=0.0)
        for token, logprobs in generate_step(
            mx.array(prompt_ids), self._model, max_tokens=self._max_tokens, sampler=sampler
        ):
            tid = int(token)
            if tid == tok.eos_token_id:
                break  # exclude the end token so decode() yields clean JSON (mirrors generate())
            tokens.append(tid)
            if not found:
                text = tok.decode(tokens)
                pos = text.find(marker)
                if pos != -1 and len(text) > pos + len(marker):
                    signal = _decision_uncertainty(
                        float(logprobs[block_first]), float(logprobs[pass_first]), mode
                    )
                    found = True
        return tok.decode(tokens).strip(), signal

    def _ensure_loaded(self) -> None:  # pragma: no cover
        if self._model is None:
            from mlx_lm import load

            loaded = load(self._model_path, **self._load_kwargs())
            self._model, self._tokenizer = loaded[0], loaded[1]
