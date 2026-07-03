# src/firewall/judge/distill/config.py
"""Typed loader for configs/distill.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class FinetunedModel:
    """A fine-tuned local judge to eval: a base model plus its LoRA adapter."""

    name: str
    base: str
    adapter_path: str


@dataclass(frozen=True)
class DistillConfig:
    """All parameters for a distillation corpus build."""

    raw_dir: Path
    output_dir: Path
    clean_threshold: float
    block_threshold: float
    classifier_path: str
    classifier_max_length: int
    generation_model: str
    teacher_model: str
    teacher_temperature: float
    target_gray_total: int
    n_generated_borderline: int
    n_generated_coercion: int
    generation_batch_size: int
    val_ratio: float
    test_ratio: float
    seed: int

    # Benign-gray top-up slice (make distill-topup). Optional/defaulted for build-only configs.
    n_generated_benign_gray: int = 0

    # --- Eval fields (Plan 4). Optional/defaulted so an existing build config still loads. ---
    reports_dir: Path = Path("reports")
    baseline_local_models: tuple[str, ...] = ()
    local_baseline_max_tokens: int = 256
    claude_price_in_per_mtok: float = 1.0
    claude_price_out_per_mtok: float = 5.0
    finetuned_local_models: tuple[FinetunedModel, ...] = ()


def load_distill_config(path: str | Path) -> DistillConfig:
    """Parse a distill YAML config into a DistillConfig."""
    data = yaml.safe_load(Path(path).read_text())
    return DistillConfig(
        raw_dir=Path(data["raw_dir"]),
        output_dir=Path(data["output_dir"]),
        clean_threshold=float(data["clean_threshold"]),
        block_threshold=float(data["block_threshold"]),
        classifier_path=str(data["classifier_path"]),
        classifier_max_length=int(data["classifier_max_length"]),
        generation_model=str(data["generation_model"]),
        teacher_model=str(data["teacher_model"]),
        teacher_temperature=float(data["teacher_temperature"]),
        target_gray_total=int(data["target_gray_total"]),
        n_generated_borderline=int(data["n_generated_borderline"]),
        n_generated_coercion=int(data["n_generated_coercion"]),
        generation_batch_size=int(data["generation_batch_size"]),
        val_ratio=float(data["val_ratio"]),
        test_ratio=float(data["test_ratio"]),
        seed=int(data["seed"]),
        n_generated_benign_gray=int(data.get("n_generated_benign_gray", 0)),
        reports_dir=Path(data.get("reports_dir", "reports")),
        baseline_local_models=tuple(data.get("baseline_local_models", ())),
        local_baseline_max_tokens=int(data.get("local_baseline_max_tokens", 256)),
        claude_price_in_per_mtok=float(data.get("claude_price_in_per_mtok", 1.0)),
        claude_price_out_per_mtok=float(data.get("claude_price_out_per_mtok", 5.0)),
        finetuned_local_models=tuple(
            FinetunedModel(name=m["name"], base=m["base"], adapter_path=m["adapter_path"])
            for m in data.get("finetuned_local_models", ())
        ),
    )
