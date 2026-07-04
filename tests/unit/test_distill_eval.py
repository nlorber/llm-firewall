from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

from firewall.judge.distill.eval import (
    JudgeRun,
    Outcome,
    build_report,
    load_test_records,
    render_markdown,
    run_all,
    tiered_summary,
)

if TYPE_CHECKING:
    from pathlib import Path


def _record(text: str, decision: str, provenance: str) -> dict[str, object]:
    return {
        "messages": [],
        "meta": {
            "text": text,
            "classifier_label": "injection",
            "scores": {"benign": 0.4, "injection": 0.5},
            "decision": decision,
            "provenance": provenance,
        },
    }


def test_run_all_injects_reference_and_provenance() -> None:
    records = [_record("a", "BLOCK", "gen"), _record("b", "PASS", "coercion")]
    outcomes = {
        "a": Outcome(predicted="BLOCK", valid=True, latency_s=0.1),
        "b": Outcome(predicted="PASS", valid=True, latency_s=0.2),
    }

    def fake_run(text: str, label: str, scores: dict[str, float]) -> Outcome:
        return outcomes[text]

    runs = run_all(fake_run, records)
    assert [r.reference for r in runs] == ["BLOCK", "PASS"]
    assert [r.provenance for r in runs] == ["gen", "coercion"]
    assert [r.predicted for r in runs] == ["BLOCK", "PASS"]


def _runs() -> list[JudgeRun]:
    # 4 records: 3 BLOCK-ref (2 hit, 1 invalid), 1 PASS-ref (hit). Two provenance slices.
    return [
        JudgeRun("BLOCK", "gen", "BLOCK", True, 0.10, 1000, 200),
        JudgeRun("BLOCK", "gen", "BLOCK", True, 0.20, 1000, 200),
        JudgeRun("BLOCK", "coercion", "INVALID", False, 0.30, 1000, 200),
        JudgeRun("PASS", "coercion", "PASS", True, 0.40, 1000, 200),
    ]


def test_build_report_core_metrics() -> None:
    report = build_report("claude", _runs(), price_in=1.0, price_out=5.0)
    assert report["name"] == "claude"
    assert report["n"] == 4
    # 3 of 4 predictions match the teacher (the INVALID one misses).
    assert math.isclose(report["decision_match"]["rate"], 0.75)
    assert report["decision_match"]["n"] == 4
    assert len(report["decision_match"]["ci"]) == 2
    # BLOCK recall: 2 of 3 reference-BLOCK predicted BLOCK.
    assert math.isclose(report["block_recall"]["rate"], 2 / 3)
    assert report["block_recall"]["n"] == 3
    # One unparseable generation.
    assert report["n_invalid"] == 1
    assert math.isclose(report["schema_validity"]["rate"], 0.75)
    # Cost per call = 1000/1e6*1 + 200/1e6*5 = 0.002.
    assert math.isclose(report["cost_per_call_usd"], 0.002)
    assert report["latency"]["p50"] > 0


def test_build_report_by_provenance_partitions() -> None:
    report = build_report("claude", _runs(), price_in=1.0, price_out=5.0)
    by_prov = report["by_provenance"]
    assert set(by_prov) == {"gen", "coercion"}
    assert by_prov["gen"]["n"] + by_prov["coercion"]["n"] == report["n"]
    # The gen slice is two clean BLOCK hits → perfect agreement.
    assert math.isclose(by_prov["gen"]["decision_match"]["rate"], 1.0)


def test_load_test_records_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    recs = [_record("a", "BLOCK", "gen"), _record("b", "PASS", "raw")]
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    loaded = load_test_records(path)
    assert [r["meta"]["decision"] for r in loaded] == ["BLOCK", "PASS"]


def test_tiered_summary_accounts_escalations() -> None:
    runs = [
        JudgeRun("BLOCK", "gen", "BLOCK", True, 0.10, tier="local", reason="none"),
        JudgeRun("BLOCK", "gen", "BLOCK", True, 0.50, tier="claude", reason="uncertainty"),
        JudgeRun("PASS", "gen", "PASS", True, 0.60, tier="claude", reason="uncertainty"),
        JudgeRun("BLOCK", "gen", "BLOCK", True, 0.20, tier="claude", reason="schema_invalid"),
    ]
    s = tiered_summary(runs, claude_cost_per_call=0.001)
    assert s["n_kept"] == 1 and s["n_escalated"] == 3
    assert math.isclose(s["escalation_rate"], 0.75)
    assert s["escalation_by_reason"] == {"uncertainty": 2, "schema_invalid": 1}
    assert math.isclose(s["blended_cost_per_call_usd"], 0.75 * 0.001)
    assert s["latency_kept"]["n"] == 1 and s["latency_escalated"]["n"] == 3


def test_render_markdown_includes_tiered_section() -> None:
    report = build_report("tiered (4B)", _runs(), price_in=0.0, price_out=0.0)
    report["tiered"] = tiered_summary(
        [JudgeRun("BLOCK", "gen", "BLOCK", True, 0.1, tier="claude", reason="uncertainty")], 0.001
    )
    md = render_markdown([report])
    assert "Tiered system" in md
    assert "escalation rate" in md


def test_render_markdown_lists_judges_and_columns() -> None:
    reports = [build_report("claude", _runs(), price_in=1.0, price_out=5.0)]
    md = render_markdown(reports)
    assert "claude" in md
    assert "Decision-match" in md
    assert "BLOCK recall" in md
    assert "Schema-valid" in md
