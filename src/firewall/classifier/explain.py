"""SHAP-based explainability for the firewall classifier.

Provides token-level attribution scores and attention heatmaps to explain
individual predictions. Used in Phase 3 and ``notebooks/03_explainability.ipynb``.
"""
from __future__ import annotations

import numpy as np


class SHAPExplainer:
    """Wraps ``shap.Explainer`` for token-level attribution on the classifier.

    Args:
        model: Loaded :class:`~firewall.classifier.model.FirewallClassifier`.
        max_evals: Maximum SHAP evaluations per example (speed vs. accuracy trade-off).
    """

    def __init__(self, model: object, max_evals: int = 500) -> None:
        raise NotImplementedError

    def explain(self, texts: list[str]) -> list[dict[str, np.ndarray]]:
        """Compute SHAP values for a batch of prompts.

        Args:
            texts: Raw prompt strings.

        Returns:
            List of dicts mapping token string → SHAP value array (one value per class).
        """
        raise NotImplementedError


def plot_attention_heatmap(
    text: str,
    model: object,
    layer: int = -1,
    head: int = 0,
) -> None:
    """Render a matplotlib attention heatmap for the given input.

    Args:
        text: Input prompt string.
        model: Loaded :class:`~firewall.classifier.model.FirewallClassifier`.
        layer: Transformer layer index to visualise (default: last layer).
        head: Attention head index.
    """
    raise NotImplementedError
