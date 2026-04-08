"""DeBERTa-v3-base classification model with custom head.

Wraps a HuggingFace AutoModelForSequenceClassification with utilities for
loading pre-trained weights, freezing backbone layers, and batch inference.
"""
from __future__ import annotations

from pathlib import Path


class FirewallClassifier:
    """Wraps a fine-tuned sequence-classification model for prompt threat detection.

    Args:
        model_name_or_path: HuggingFace model ID or local checkpoint directory.
        num_labels: Number of threat categories (default 5).
        device: Torch device string. Defaults to CUDA when available.
    """

    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int = 5,
        device: str | None = None,
    ) -> None:
        raise NotImplementedError

    def predict(self, texts: list[str]) -> list[dict[str, float]]:
        """Run batch inference and return per-class probability dicts.

        Args:
            texts: Raw prompt strings.

        Returns:
            List of dicts mapping label name → probability, one per input text.
        """
        raise NotImplementedError

    def save(self, output_dir: str | Path) -> None:
        """Persist model weights and tokenizer to *output_dir*.

        Args:
            output_dir: Destination directory (created if absent).
        """
        raise NotImplementedError


def load_classifier(checkpoint_path: str | Path) -> FirewallClassifier:
    """Load a fine-tuned :class:`FirewallClassifier` from a checkpoint directory.

    Args:
        checkpoint_path: Path to a directory containing ``config.json`` and weights.

    Returns:
        Loaded classifier placed in evaluation mode.
    """
    raise NotImplementedError
