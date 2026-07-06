from __future__ import annotations

from pathlib import Path

import yaml

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_gray_thresholds_match_across_configs() -> None:
    """distill.yaml and orchestrator.yaml both carry the GRAY band, and their comments say the
    thresholds MUST match — the corpus has to be built off the same band the live judge routes
    on. Enforce it so drift can't silently build an off-distribution corpus.
    """
    distill = yaml.safe_load((_CONFIGS / "distill.yaml").read_text())
    orch = yaml.safe_load((_CONFIGS / "orchestrator.yaml").read_text())
    assert distill["clean_threshold"] == orch["clean_threshold"]
    assert distill["block_threshold"] == orch["block_threshold"]
