# src/firewall/judge/distill/staleness.py
"""Staleness probe — ground-truth BLOCK recall on the held-out adversarial set.

Distinct from the teacher-agreement eval: every prompt in ``data/adversarial/`` is a *real*
attack, so BLOCK is the only correct call and recall is measured against ground truth, not
against Claude. We keep only the GRAY-band subset (the attacks the classifier is unsure about
— the judge's actual domain) and report that N honestly; it is expected to be small. This is
the independent safety read the distilled/tiered judge must survive.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from firewall.judge.base import Judge
    from firewall.judge.distill.data import GrayCandidate


def ground_truth_block_recall(decisions: list[str]) -> tuple[float, int]:
    """Fraction of adversarial prompts the judge BLOCKed (all are attacks → BLOCK is correct).

    Returns ``(recall, n)``; empty input yields ``(0.0, 0)``. A missed BLOCK here is a real
    false PASS of a known attack — the safety failure the teacher-agreement metric cannot see.
    """
    if not decisions:
        return (0.0, 0)
    blocked = sum(1 for d in decisions if d == "BLOCK")
    return (blocked / len(decisions), len(decisions))


def run_judge_on_gray(judge: Judge, gray: list[GrayCandidate]) -> list[str]:
    """Judge each GRAY adversarial candidate, feeding the same (label, scores) the live judge
    gets, and return the decisions."""
    return [judge.judge(g.text, g.classifier_label, g.scores).decision for g in gray]


def load_adversarial_texts(path: Path) -> list[str]:
    """Read the adversarial prompts (JSONL with a ``text`` field)."""
    return [json.loads(line)["text"] for line in path.read_text().splitlines() if line.strip()]


def main() -> None:  # pragma: no cover
    """GRAY-filter the adversarial set, then report ground-truth BLOCK recall per judge."""
    import argparse
    from pathlib import Path

    from dotenv import load_dotenv

    from firewall.classifier.model import load_classifier
    from firewall.judge.distill.config import load_distill_config
    from firewall.judge.distill.data import Candidate, classify_and_filter_gray
    from firewall.judge.tiered import make_judge

    parser = argparse.ArgumentParser(
        description="Ground-truth staleness probe on data/adversarial"
    )
    parser.add_argument("--config", default="configs/distill.yaml", type=Path)
    parser.add_argument(
        "--adversarial", default="data/adversarial/adversarial_prompts.jsonl", type=Path
    )
    args = parser.parse_args()

    load_dotenv()
    cfg = load_distill_config(args.config)
    classifier = load_classifier(cfg.classifier_path, max_length=cfg.classifier_max_length)

    texts = load_adversarial_texts(args.adversarial)
    candidates = [Candidate(text=t, provenance="adversarial") for t in texts]
    gray = classify_and_filter_gray(
        classifier, candidates, cfg.clean_threshold, cfg.block_threshold
    )
    print(f"[staleness] adversarial: {len(texts)} total → {len(gray)} in GRAY band (the probe N)")
    if not gray:
        print("[staleness] no adversarial prompts land in GRAY — the classifier blocks them all.")
        return

    judges: dict[str, Judge] = {"claude": make_judge("claude", teacher_model=cfg.teacher_model)}
    ft = next((f for f in cfg.finetuned_local_models if f.name == cfg.tiered_model_name), None)
    if ft is not None and Path(ft.adapter_path).exists():
        judges[f"local {ft.name}"] = make_judge(
            "local",
            local_model=ft.base,
            adapter_path=ft.adapter_path,
            max_tokens=cfg.local_baseline_max_tokens,
        )
        judges["tiered"] = make_judge(
            "tiered",
            teacher_model=cfg.teacher_model,
            local_model=ft.base,
            adapter_path=ft.adapter_path,
            threshold=cfg.tiered_threshold,
            max_tokens=cfg.local_baseline_max_tokens,
        )

    for name, judge in judges.items():
        recall, n = ground_truth_block_recall(run_judge_on_gray(judge, gray))
        print(f"[staleness] {name:28} ground-truth BLOCK recall: {recall * 100:.0f}% (n={n})")


if __name__ == "__main__":  # pragma: no cover
    main()
