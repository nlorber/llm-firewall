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


def test_eval_fields_default_when_absent(tmp_path: Path) -> None:
    # A build-only config (no eval keys) still loads; eval fields fall back to defaults.
    path = tmp_path / "distill.yaml"
    path.write_text(
        "raw_dir: data/raw\noutput_dir: data/distill\nclean_threshold: 0.3\n"
        "block_threshold: 0.8\nclassifier_path: models/classifier\nclassifier_max_length: 512\n"
        "generation_model: m\nteacher_model: m\nteacher_temperature: 0.0\n"
        "target_gray_total: 10\nn_generated_borderline: 5\nn_generated_coercion: 5\n"
        "generation_batch_size: 5\nval_ratio: 0.15\ntest_ratio: 0.15\nseed: 42\n"
    )
    cfg = load_distill_config(path)
    assert cfg.reports_dir == Path("reports")
    assert cfg.baseline_local_models == ()
    assert cfg.local_baseline_max_tokens == 256
    assert cfg.claude_price_in_per_mtok == 1.0


def test_eval_fields_parsed_when_present(tmp_path: Path) -> None:
    path = tmp_path / "distill.yaml"
    path.write_text(
        "raw_dir: data/raw\noutput_dir: data/distill\nclean_threshold: 0.3\n"
        "block_threshold: 0.8\nclassifier_path: models/classifier\nclassifier_max_length: 512\n"
        "generation_model: m\nteacher_model: m\nteacher_temperature: 0.0\n"
        "target_gray_total: 10\nn_generated_borderline: 5\nn_generated_coercion: 5\n"
        "generation_batch_size: 5\nval_ratio: 0.15\ntest_ratio: 0.15\nseed: 42\n"
        'reports_dir: out\nbaseline_local_models: ["a", "b"]\nlocal_baseline_max_tokens: 128\n'
        "claude_price_in_per_mtok: 0.8\nclaude_price_out_per_mtok: 4.0\n"
    )
    cfg = load_distill_config(path)
    assert cfg.reports_dir == Path("out")
    assert cfg.baseline_local_models == ("a", "b")
    assert cfg.local_baseline_max_tokens == 128
    assert cfg.claude_price_out_per_mtok == 4.0
