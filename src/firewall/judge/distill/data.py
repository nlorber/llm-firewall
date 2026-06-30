# src/firewall/judge/distill/data.py
"""Build the distillation corpus: source -> classify/GRAY-filter -> teacher-label ->
assemble role-tagged SFT records -> stratified split.

Pure steps take injected dependencies (classifier, judge, Claude client) so the whole
pipeline is unit-tested with fakes; the CLI wires the real ones.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path


class _Classifier(Protocol):
    def predict(self, prompts: list[str]) -> list[dict[str, float]]: ...


@dataclass
class Candidate:
    """A sourced prompt awaiting classification."""

    text: str
    provenance: str  # "raw" | "gen" | "coercion"


@dataclass
class GrayCandidate:
    """A candidate the classifier placed in the GRAY band."""

    text: str
    provenance: str
    classifier_label: str
    scores: dict[str, float]
    threat_score: float


def load_raw_candidates(raw_dir: Path) -> list[Candidate]:
    """Load all *.jsonl under raw_dir as `raw`-provenance candidates (deduped by text)."""
    seen: set[str] = set()
    out: list[Candidate] = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            text = str(json.loads(line)["text"]).strip()
            if text and text not in seen:
                seen.add(text)
                out.append(Candidate(text=text, provenance="raw"))
    return out


def _threat_score(scores: dict[str, float]) -> float:
    """Max non-benign probability — the value the orchestrator routes on."""
    return max((v for k, v in scores.items() if k != "benign"), default=0.0)


def classify_and_filter_gray(
    classifier: _Classifier,
    candidates: list[Candidate],
    clean_threshold: float,
    block_threshold: float,
) -> list[GrayCandidate]:
    """Classify candidates and keep only those whose threat score is in the GRAY band."""
    if not candidates:
        return []
    score_dicts = classifier.predict([c.text for c in candidates])
    gray: list[GrayCandidate] = []
    for cand, scores in zip(candidates, score_dicts, strict=True):
        threat = _threat_score(scores)
        if clean_threshold <= threat < block_threshold:
            top_label = max(scores, key=scores.__getitem__)
            gray.append(
                GrayCandidate(
                    text=cand.text,
                    provenance=cand.provenance,
                    classifier_label=top_label,
                    scores=scores,
                    threat_score=threat,
                )
            )
    return gray
