"""Evaluation utilities: metrics, confusion matrix, per-class report.

Entry point::

    python -m firewall.classifier.evaluate --config configs/training.yaml
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def evaluate(
    model_path: str | Path,
    test_path: str | Path,
    label_names: list[str],
) -> dict[str, float | np.ndarray]:
    """Run inference on the test split and compute full evaluation metrics.

    Args:
        model_path: Path to a fine-tuned checkpoint directory.
        test_path: Path to ``data/processed/test.jsonl``.
        label_names: Ordered list of class names for the classification report.

    Returns:
        Dict containing ``accuracy``, ``f1_macro``, ``f1_weighted``,
        ``confusion_matrix`` (ndarray), and ``classification_report`` (str).
    """
    raise NotImplementedError


def main() -> None:
    """CLI entry point: parse ``--config`` argument and call :func:`evaluate`."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
