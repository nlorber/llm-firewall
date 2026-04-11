# src/firewall/classifier/explain.py
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import shap

if TYPE_CHECKING:
    from firewall.classifier.model import FirewallClassifier


_DEFAULT_MAX_EVALS = 500
_MIN_FIGSIZE = 6
_FIGSIZE_DIVISOR = 2
_TICK_FONTSIZE = 8
_SAVE_DPI = 150


class SHAPExplainer:
    """Token-level SHAP attributions for the firewall classifier."""

    def __init__(self, model: Any, max_evals: int = _DEFAULT_MAX_EVALS) -> None:
        self._model = model

        def _predict_proba(texts: list[str]) -> np.ndarray[Any, np.dtype[Any]]:
            results = model.predict(list(texts))
            return np.array([[v for v in r.values()] for r in results])

        self._explainer = shap.Explainer(
            _predict_proba,
            shap.maskers.Text(r"\W+"),
            max_evals=max_evals,
        )

    def explain(self, texts: list[str]) -> list[dict[str, Any]]:
        """Return SHAP values and tokens for each text.

        Returns:
            List of dicts with keys ``tokens`` (array of token strings) and
            ``shap_values`` (ndarray shape [n_tokens, n_classes]).
        """
        shap_values = self._explainer(texts)
        results = []
        for i in range(len(texts)):
            results.append({
                "tokens":      shap_values.data[i],
                "shap_values": shap_values.values[i],
            })
        return results


def plot_attention_heatmap(
    text: str,
    model: FirewallClassifier,
    layer: int = -1,
    head: int = 0,
    output_path: str | Path = "attention_heatmap.png",
) -> Path:
    """Render a matplotlib attention heatmap for a single input.

    Returns:
        Path to the saved figure.
    """
    import matplotlib.pyplot as plt
    import torch

    tokenizer = model.tokenizer
    tokens = tokenizer.tokenize(text)

    encoding = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.model(**encoding, output_attentions=True)

    attn = outputs.attentions[layer][0, head].cpu().numpy()

    n = len(tokens)
    fig, ax = plt.subplots(figsize=(max(_MIN_FIGSIZE, n // _FIGSIZE_DIVISOR), max(_MIN_FIGSIZE, n // _FIGSIZE_DIVISOR)))
    im = ax.imshow(attn[:n, :n], cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tokens, rotation=90, fontsize=_TICK_FONTSIZE)
    ax.set_yticklabels(tokens, fontsize=_TICK_FONTSIZE)
    ax.set_title(f"Attention layer {layer} head {head}")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    out = Path(output_path)
    fig.savefig(out, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out
