# src/firewall/judge/distill/eval.py
"""Distillation eval harness — the 'before' baselines run *before* any training.

Per-judge metrics only (teacher-agreement decision-match + BLOCK recall with N + Wilson
CIs, JSON schema-validity, latency, cost, by-provenance breakdown). The composed
tiered-system metrics (escalation rate, blended cost, signal AUC) are deferred to the
training stage, where a TieredJudge and decision-token signals exist to produce them.

The compute path — :func:`run_all`, :func:`build_report`, :func:`render_markdown` — is
pure and unit-tested with fake run functions; the two adapters that drive the real Claude
and MLX judges are the only model-touching code and are excluded from unit coverage.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from firewall.judge.base import parse_verdict
from firewall.judge.distill.metrics import (
    benign_pass_rate,
    block_recall,
    confusion_counts,
    count_invalid,
    decision_match,
    latency_stats,
    schema_validity_rate,
    token_cost_usd,
    wilson_ci,
)
from firewall.judge.local_judge import ThinkingModeError

if TYPE_CHECKING:
    from pathlib import Path

    from firewall.judge.judge import LLMJudge
    from firewall.judge.local_judge import LocalJudge

_THINK_OPEN = "<think>"
_INVALID = "INVALID"


@dataclass
class Outcome:
    """What one judge call produced, independent of the record it ran on."""

    predicted: str  # "PASS" | "BLOCK" | "INVALID"
    valid: bool
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


@dataclass
class JudgeRun:
    """An :class:`Outcome` paired with the teacher reference + provenance of its record."""

    reference: str
    provenance: str
    predicted: str
    valid: bool
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


RunFn = Callable[[str, str, dict[str, float]], Outcome]


def run_all(run_fn: RunFn, records: list[dict[str, Any]]) -> list[JudgeRun]:
    """Run ``run_fn`` over every record, attaching each record's teacher label + provenance."""
    runs: list[JudgeRun] = []
    for rec in records:
        meta = rec["meta"]
        out = run_fn(meta["text"], meta["classifier_label"], meta["scores"])
        runs.append(
            JudgeRun(
                reference=meta["decision"],
                provenance=meta["provenance"],
                predicted=out.predicted,
                valid=out.valid,
                latency_s=out.latency_s,
                input_tokens=out.input_tokens,
                output_tokens=out.output_tokens,
                error=out.error,
            )
        )
    return runs


def _ci(rate: float, n: int) -> list[float]:
    """Wilson interval for a rate observed over n trials (successes recovered as rate*n)."""
    low, high = wilson_ci(round(rate * n), n)
    return [low, high]


def _core_metrics(runs: list[JudgeRun], price_in: float, price_out: float) -> dict[str, Any]:
    """The per-judge metric block, computed over a full set or a provenance slice."""
    pairs = [(r.predicted, r.reference) for r in runs]
    dm_rate, dm_n = decision_match(pairs)
    br_rate, br_n = block_recall(pairs)
    bp_rate, bp_n = benign_pass_rate(pairs)
    sv_rate, sv_n = schema_validity_rate([r.valid for r in runs])
    costs = [token_cost_usd(r.input_tokens, r.output_tokens, price_in, price_out) for r in runs]
    return {
        "n": len(runs),
        "decision_match": {"rate": dm_rate, "ci": _ci(dm_rate, dm_n), "n": dm_n},
        "block_recall": {"rate": br_rate, "ci": _ci(br_rate, br_n), "n": br_n},
        "benign_pass_rate": {"rate": bp_rate, "ci": _ci(bp_rate, bp_n), "n": bp_n},
        "schema_validity": {"rate": sv_rate, "n": sv_n},
        "confusion": confusion_counts(pairs),
        "n_invalid": count_invalid(pairs),
        "latency": latency_stats([r.latency_s for r in runs]),
        "cost_per_call_usd": sum(costs) / len(costs) if costs else 0.0,
        "n_errors": sum(1 for r in runs if r.error),
    }


def build_report(
    name: str, runs: list[JudgeRun], price_in: float, price_out: float
) -> dict[str, Any]:
    """Assemble the full report for one judge: overall metrics + a by-provenance breakdown.

    The by-provenance split matters because Claude judging Claude-generated prompts is
    partly self-consistency; the honest reference is the non-Claude-generated slice.
    """
    report: dict[str, Any] = {"name": name, **_core_metrics(runs, price_in, price_out)}
    by_prov: dict[str, Any] = {}
    for prov in sorted({r.provenance for r in runs}):
        by_prov[prov] = _core_metrics(
            [r for r in runs if r.provenance == prov], price_in, price_out
        )
    report["by_provenance"] = by_prov
    return report


def load_test_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a distillation split (JSONL, one record per line)."""
    from pathlib import Path as _Path

    lines = _Path(path).read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _fmt_pct(block: dict[str, Any]) -> str:
    lo, hi = block["ci"]
    return f"{block['rate'] * 100:.1f}% [{lo * 100:.0f}–{hi * 100:.0f}] (n={block['n']})"


def render_markdown(reports: list[dict[str, Any]]) -> str:
    """Render the 'before' comparison table (+ a by-provenance section) as markdown."""
    lines = [
        "| Judge | N | Decision-match | BLOCK recall | Benign-PASS | Schema-valid "
        "| Latency p50/p95 (s) | Cost/call |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        lat = r["latency"]
        sv = r["schema_validity"]
        lines.append(
            f"| {r['name']} | {r['n']} | {_fmt_pct(r['decision_match'])} "
            f"| {_fmt_pct(r['block_recall'])} | {_fmt_pct(r['benign_pass_rate'])} "
            f"| {sv['rate'] * 100:.1f}% | {lat['p50']:.3f} / {lat['p95']:.3f} "
            f"| ${r['cost_per_call_usd']:.5f} |"
        )
    lines.append("")
    lines.append("### Agreement by provenance (same-family generation inflates agreement)")
    for r in reports:
        lines.append(f"\n**{r['name']}**")
        lines.append("| Provenance | N | Decision-match | BLOCK recall |")
        lines.append("|---|---|---|---|")
        for prov, block in r["by_provenance"].items():
            lines.append(
                f"| {prov} | {block['n']} | {_fmt_pct(block['decision_match'])} "
                f"| {_fmt_pct(block['block_recall'])} |"
            )
    return "\n".join(lines)


# ---- Model-touching adapters (real judges) — excluded from unit coverage ----------------


def run_claude_fn(judge: LLMJudge) -> RunFn:  # pragma: no cover
    """A RunFn that times ``judge_verbose`` and records real token usage."""

    def _run(text: str, label: str, scores: dict[str, float]) -> Outcome:
        start = time.perf_counter()
        try:
            verdict, _raw, usage = judge.judge_verbose(text, label, scores)
        except Exception as exc:  # noqa: BLE001 — one bad record must not abort the sweep
            return Outcome(_INVALID, False, time.perf_counter() - start, error=str(exc))
        return Outcome(
            predicted=verdict.decision,
            valid=True,
            latency_s=time.perf_counter() - start,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
        )

    return _run


def run_local_fn(judge: LocalJudge) -> RunFn:  # pragma: no cover
    """A RunFn that times ``generate_raw`` and parses without repair.

    A leaked ``<think>`` block aborts the whole run (hard assertion): it means
    schema-validity would be measuring thinking-leak, not model quality.
    """

    def _run(text: str, label: str, scores: dict[str, float]) -> Outcome:
        start = time.perf_counter()
        raw = judge.generate_raw(text, label, scores)
        latency = time.perf_counter() - start
        if _THINK_OPEN in raw:
            raise ThinkingModeError(
                "local model emitted a <think> block during eval; use a non-thinking model"
            )
        try:
            verdict = parse_verdict(raw)
        except (ValueError, KeyError):
            return Outcome(_INVALID, False, latency)
        return Outcome(predicted=verdict.decision, valid=True, latency_s=latency)

    return _run


def main() -> None:  # pragma: no cover
    """Run the 'before' baselines (Claude reference + base MLX models) and write reports/."""
    import argparse
    from datetime import date
    from pathlib import Path

    from dotenv import load_dotenv

    from firewall.judge.distill.config import load_distill_config
    from firewall.judge.judge import LLMJudge
    from firewall.judge.local_judge import LocalJudge

    parser = argparse.ArgumentParser(description="Eval distillation baselines (before training)")
    parser.add_argument("--config", default="configs/distill.yaml", type=Path)
    args = parser.parse_args()

    load_dotenv()  # ANTHROPIC_API_KEY for the Claude reference row
    config = load_distill_config(args.config)
    records = load_test_records(config.output_dir / "test.jsonl")
    print(f"[distill-eval] {len(records)} test records from {config.output_dir / 'test.jsonl'}")

    reports: list[dict[str, Any]] = []

    # Claude teacher = reference row (agreement ~1.0 by construction; a temp-0 re-run also
    # sanity-checks teacher determinism and gives the real latency/cost numbers).
    print("[distill-eval] running Claude reference...")
    claude = LLMJudge(model=config.teacher_model, temperature=config.teacher_temperature)
    claude_runs = run_all(run_claude_fn(claude), records)
    reports.append(
        build_report(
            f"claude ({config.teacher_model})",
            claude_runs,
            config.claude_price_in_per_mtok,
            config.claude_price_out_per_mtok,
        )
    )

    # Base (un-fine-tuned) local models, prompted — the baseline the fine-tune must beat.
    for model_path in config.baseline_local_models:
        print(f"[distill-eval] running base local model {model_path}...")
        local = LocalJudge(model_path, max_tokens=config.local_baseline_max_tokens)
        local_runs = run_all(run_local_fn(local), records)
        reports.append(build_report(f"base {model_path}", local_runs, 0.0, 0.0))

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    json_path = config.reports_dir / f"distill_eval_baselines_{stamp}.json"
    md_path = config.reports_dir / f"distill_eval_baselines_{stamp}.md"
    json_path.write_text(json.dumps(reports, indent=2))
    markdown = render_markdown(reports)
    md_path.write_text(markdown + "\n")
    print(f"\n{markdown}\n\n[distill-eval] wrote {json_path} and {md_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
