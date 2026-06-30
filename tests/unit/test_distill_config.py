from __future__ import annotations

from pathlib import Path

from firewall.judge.distill.config import DistillConfig, load_distill_config


def test_loads_all_fields(tmp_path: Path) -> None:
    cfg_text = """
raw_dir: data/raw
output_dir: data/distill
clean_threshold: 0.3
block_threshold: 0.8
classifier_path: models/classifier
classifier_max_length: 512
generation_model: claude-haiku-4-5-20251001
teacher_model: claude-haiku-4-5-20251001
teacher_temperature: 0.0
target_gray_total: 1200
n_generated_borderline: 1500
n_generated_coercion: 600
generation_batch_size: 20
val_ratio: 0.15
test_ratio: 0.15
seed: 42
"""
    path = tmp_path / "distill.yaml"
    path.write_text(cfg_text)
    cfg = load_distill_config(path)
    assert isinstance(cfg, DistillConfig)
    assert cfg.raw_dir == Path("data/raw")
    assert cfg.clean_threshold == 0.3
    assert cfg.target_gray_total == 1200
    assert cfg.teacher_temperature == 0.0
    assert cfg.seed == 42
