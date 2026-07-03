# src/firewall/judge/distill/train.py
"""Fine-tune a base Qwen3 on the distillation corpus (QLoRA via mlx-lm).

The data-prep step here is pure and unit-tested: it turns our role-tagged split records
into the messages-only JSONL that mlx-lm's chat dataset expects, with loss taken only on
the assistant completion (``mask_prompt: true`` in the config). The actual training is a
subprocess to ``mlx_lm.lora`` (Apple-Silicon only) and is excluded from unit coverage.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

# mlx-lm's data dir must hold `train.jsonl` + `valid.jsonl`; our splits are train/val/test.
_SPLIT_TO_MLX = {"train": "train", "val": "valid"}


def to_mlx_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the role-tagged ``messages`` — mlx-lm reads that key; ``meta`` is for eval."""
    return [{"messages": r["messages"]} for r in records]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare_mlx_data(src_dir: Path, out_dir: Path) -> dict[str, int]:
    """Convert our corpus splits into an mlx-lm data dir (messages-only train/valid.jsonl)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split, mlx_name in _SPLIT_TO_MLX.items():
        records = _read_jsonl(src_dir / f"{split}.jsonl")
        mlx = to_mlx_records(records)
        (out_dir / f"{mlx_name}.jsonl").write_text("\n".join(json.dumps(m) for m in mlx) + "\n")
        counts[mlx_name] = len(mlx)
    return counts


def main() -> None:  # pragma: no cover
    """Prepare the mlx-lm data dir (from the config's ``data:``) and run ``mlx_lm.lora``."""
    import argparse
    import subprocess
    import sys
    from pathlib import Path

    import yaml

    parser = argparse.ArgumentParser(description="Data-prep + QLoRA-train a judge via mlx-lm")
    parser.add_argument("--config", required=True, type=Path, help="mlx-lm LoRA YAML config")
    parser.add_argument("--data-src", default=Path("data/distill"), type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    out_dir = Path(cfg["data"])
    counts = prepare_mlx_data(args.data_src, out_dir)
    print(f"[distill-train] prepared mlx data at {out_dir}: {counts}")

    cmd = [sys.executable, "-m", "mlx_lm", "lora", "--config", str(args.config)]
    print(f"[distill-train] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":  # pragma: no cover
    main()
