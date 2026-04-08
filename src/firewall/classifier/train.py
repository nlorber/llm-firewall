"""HuggingFace Trainer fine-tuning loop for the firewall classifier.

Entry point::

    python -m firewall.classifier.train --config configs/training.yaml

Responsibilities:
- Load train/val splits via :mod:`firewall.classifier.dataset`
- Instantiate :class:`~firewall.classifier.model.FirewallClassifier`
- Configure HF Trainer with class-weighted loss, early stopping, and checkpointing
- Save the best checkpoint to ``config.output_dir``
"""
from __future__ import annotations

from pathlib import Path


def train(config_path: str | Path) -> None:
    """Run the full fine-tuning loop from a YAML config file.

    Args:
        config_path: Path to ``configs/training.yaml``.
    """
    raise NotImplementedError


def compute_metrics(eval_pred: tuple) -> dict[str, float]:
    """Compute accuracy, macro-F1, and weighted-F1 for the HF Trainer callback.

    Args:
        eval_pred: ``(logits, labels)`` tuple produced by the Trainer.

    Returns:
        Dict with keys ``accuracy``, ``f1_macro``, ``f1_weighted``.
    """
    raise NotImplementedError


def main() -> None:
    """CLI entry point: parse ``--config`` argument and call :func:`train`."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
