# src/firewall/judge/distill/metrics.py
"""Pure metric primitives for the distillation eval harness.

Stdlib-only and side-effect-free, so the whole scoring layer is unit-tested without a
model or the Anthropic SDK (it is the coverage backbone for the MLX-gated eval).

**Teacher-agreement framing.** The reference decision is the Claude teacher's stored
label, so every rate here is *agreement with the teacher*, never a ground-truth
guarantee — a teacher miss is invisible. BLOCK is the positive class: a missed BLOCK is
a false PASS, the costly firewall error, which is why BLOCK recall is a headline metric.

A schema-invalid generation carries the sentinel decision ``"INVALID"``: it flows through
the pair-based metrics as a non-matching label (so it correctly lowers match/recall),
while :func:`schema_validity_rate` reports separately how often that happened.
"""

from __future__ import annotations

import math

BLOCK = "BLOCK"
PASS = "PASS"

DecisionPair = tuple[str, str]  # (predicted, reference)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation at small n / extreme p — the test split has
    only ~6 PASS positives, where the normal interval spills outside [0, 1] and reads far
    too tight. Returns ``(low, high)``; ``n == 0`` yields ``(0.0, 0.0)``.
    """
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return ((center - margin) / denom, (center + margin) / denom)


def decision_match(pairs: list[DecisionPair]) -> tuple[float, int]:
    """Overall teacher-agreement: fraction of pairs where predicted == reference.

    Returns ``(rate, n)``. An ``INVALID`` prediction never matches, so it counts as a miss.
    """
    if not pairs:
        return (0.0, 0)
    hits = sum(1 for pred, ref in pairs if pred == ref)
    return (hits / len(pairs), len(pairs))


def _recall(pairs: list[DecisionPair], positive: str) -> tuple[float, int]:
    """Of pairs whose reference is ``positive``, the fraction also predicted ``positive``."""
    relevant = [pred for pred, ref in pairs if ref == positive]
    if not relevant:
        return (0.0, 0)
    hits = sum(1 for pred in relevant if pred == positive)
    return (hits / len(relevant), len(relevant))


def block_recall(pairs: list[DecisionPair]) -> tuple[float, int]:
    """Sensitivity: of reference==BLOCK, fraction predicted BLOCK. Returns ``(rate, n_pos)``."""
    return _recall(pairs, BLOCK)


def benign_pass_rate(pairs: list[DecisionPair]) -> tuple[float, int]:
    """Specificity: of reference==PASS, fraction predicted PASS. Returns ``(rate, n_neg)``."""
    return _recall(pairs, PASS)


def confusion_counts(pairs: list[DecisionPair]) -> dict[str, int]:
    """Confusion cells with BLOCK as the positive class, over valid PASS/BLOCK preds only.

    Invalid predictions are excluded here and tallied by :func:`count_invalid`, so the four
    cells stay a clean 2x2 and do not silently absorb schema failures.
    """
    tp = fp = tn = fn = 0
    for pred, ref in pairs:
        if pred == BLOCK and ref == BLOCK:
            tp += 1
        elif pred == BLOCK and ref == PASS:
            fp += 1
        elif pred == PASS and ref == PASS:
            tn += 1
        elif pred == PASS and ref == BLOCK:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def count_invalid(pairs: list[DecisionPair]) -> int:
    """Number of predictions that are neither PASS nor BLOCK (schema-invalid generations)."""
    return sum(1 for pred, _ in pairs if pred not in {PASS, BLOCK})


def schema_validity_rate(valid_flags: list[bool]) -> tuple[float, int]:
    """Fraction of generations that parsed to a valid verdict. Returns ``(rate, n)``."""
    if not valid_flags:
        return (0.0, 0)
    return (sum(valid_flags) / len(valid_flags), len(valid_flags))


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile (``q`` in [0, 100]) matching numpy's default method."""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (q / 100.0) * (len(sorted_vals) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_vals[lo]
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def latency_stats(latencies: list[float]) -> dict[str, float]:
    """p50 / p95 / mean (seconds) plus n; empty input returns zeros without raising."""
    if not latencies:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0, "n": 0}
    ordered = sorted(latencies)
    return {
        "p50": _percentile(ordered, 50),
        "p95": _percentile(ordered, 95),
        "mean": sum(ordered) / len(ordered),
        "n": len(ordered),
    }


def auc(scores: list[float], labels: list[bool]) -> float:
    """Area under the ROC curve: P(a positive's score > a negative's), ties counting 0.5.

    Used to rank escalation-signal candidates by how well they predict local-judge
    disagreement-with-teacher on val. 0.5 = no discriminative power; returns 0.5 when a class
    is absent (undefined). O(n^2), fine for the small val split.
    """
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1.0
            elif a == b:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def token_cost_usd(
    input_tokens: int,
    output_tokens: int,
    price_in_per_mtok: float,
    price_out_per_mtok: float,
) -> float:
    """Dollar cost of one call from token usage and per-million-token prices (local ≈ $0)."""
    return input_tokens / 1e6 * price_in_per_mtok + output_tokens / 1e6 * price_out_per_mtok
