from __future__ import annotations

import json
from typing import TYPE_CHECKING

from firewall.judge.base import JudgeVerdict
from firewall.judge.distill.data import GrayCandidate
from firewall.judge.distill.staleness import (
    ground_truth_block_recall,
    load_adversarial_texts,
    run_judge_on_gray,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_ground_truth_block_recall() -> None:
    # All adversarial → BLOCK is correct; recall = fraction blocked.
    assert ground_truth_block_recall(["BLOCK", "BLOCK", "PASS", "BLOCK"]) == (0.75, 4)
    assert ground_truth_block_recall(["BLOCK", "BLOCK"]) == (1.0, 2)
    assert ground_truth_block_recall([]) == (0.0, 0)


class _FakeJudge:
    def __init__(self, table: dict[str, str]) -> None:
        self._table = table

    def judge(
        self, prompt: str, classification_label: str, scores: dict[str, float]
    ) -> JudgeVerdict:
        return JudgeVerdict(self._table[prompt], "r", 0.7)


def test_run_judge_on_gray_returns_decisions() -> None:
    gray = [
        GrayCandidate("attack1", "adversarial", "injection", {"injection": 0.5}, 0.5),
        GrayCandidate("attack2", "adversarial", "jailbreak", {"jailbreak": 0.6}, 0.6),
    ]
    judge = _FakeJudge({"attack1": "BLOCK", "attack2": "PASS"})
    assert run_judge_on_gray(judge, gray) == ["BLOCK", "PASS"]


def test_load_adversarial_texts(tmp_path: Path) -> None:
    path = tmp_path / "adv.jsonl"
    path.write_text(
        json.dumps({"text": "a", "label": "injection"})
        + "\n"
        + json.dumps({"text": "b", "label": "exfiltration"})
        + "\n"
    )
    assert load_adversarial_texts(path) == ["a", "b"]
