from __future__ import annotations

import math

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


def test_wilson_ci_known_values() -> None:
    # Symmetric case p=0.5, n=100 — center 0.5, roughly +/-0.098.
    lo, hi = wilson_ci(50, 100)
    assert math.isclose((lo + hi) / 2, 0.5, abs_tol=0.02)
    assert 0.39 < lo < 0.41 and 0.59 < hi < 0.61
    # Extreme p with small n stays inside [0, 1] (where the normal approx would not).
    lo0, hi0 = wilson_ci(0, 6)
    assert lo0 == 0.0
    assert 0.0 < hi0 < 0.45
    lo1, hi1 = wilson_ci(6, 6)
    assert hi1 == 1.0
    assert 0.5 < lo1 < 1.0
    # Zero-n never raises.
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_decision_match_counts_invalid_as_miss() -> None:
    pairs = [
        ("BLOCK", "BLOCK"),  # hit
        ("PASS", "PASS"),  # hit
        ("PASS", "BLOCK"),  # miss
        ("INVALID", "BLOCK"),  # miss (unparseable)
    ]
    rate, n = decision_match(pairs)
    assert n == 4
    assert math.isclose(rate, 0.5)
    assert decision_match([]) == (0.0, 0)


def test_block_recall_isolates_block_reference() -> None:
    pairs = [
        ("BLOCK", "BLOCK"),  # positive, hit
        ("PASS", "BLOCK"),  # positive, miss
        ("INVALID", "BLOCK"),  # positive, miss
        ("PASS", "PASS"),  # negative — ignored
    ]
    rate, n_pos = block_recall(pairs)
    assert n_pos == 3
    assert math.isclose(rate, 1 / 3)


def test_benign_pass_rate_isolates_pass_reference() -> None:
    pairs = [
        ("PASS", "PASS"),  # negative, hit
        ("BLOCK", "PASS"),  # negative, miss
        ("BLOCK", "BLOCK"),  # positive — ignored
    ]
    rate, n_neg = benign_pass_rate(pairs)
    assert n_neg == 2
    assert math.isclose(rate, 0.5)


def test_confusion_counts_and_invalid() -> None:
    pairs = [
        ("BLOCK", "BLOCK"),  # tp
        ("BLOCK", "PASS"),  # fp
        ("PASS", "PASS"),  # tn
        ("PASS", "BLOCK"),  # fn
        ("INVALID", "BLOCK"),  # not counted in confusion; is invalid
    ]
    conf = confusion_counts(pairs)
    assert conf == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    assert count_invalid(pairs) == 1


def test_schema_validity_rate() -> None:
    assert schema_validity_rate([True, True, False, True]) == (0.75, 4)
    assert schema_validity_rate([]) == (0.0, 0)


def test_latency_stats_percentiles() -> None:
    stats = latency_stats([0.1, 0.2, 0.3, 0.4, 0.5])
    assert math.isclose(stats["p50"], 0.3)
    assert math.isclose(stats["mean"], 0.3)
    # p95 of 5 points via linear interpolation between 0.4 and 0.5.
    assert 0.4 < stats["p95"] <= 0.5
    assert stats["n"] == 5
    empty = latency_stats([])
    assert empty["p50"] == 0.0 and empty["n"] == 0
    # Single sample: every percentile is that sample.
    one = latency_stats([0.42])
    assert one["p50"] == 0.42 and one["p95"] == 0.42 and one["mean"] == 0.42


def test_token_cost_usd() -> None:
    # 1000 in @ $1/MTok + 200 out @ $5/MTok = 0.001 + 0.001 = 0.002
    cost = token_cost_usd(1000, 200, price_in_per_mtok=1.0, price_out_per_mtok=5.0)
    assert math.isclose(cost, 0.002)
    assert token_cost_usd(0, 0, 1.0, 5.0) == 0.0
