# src/firewall/classifier/explain.py
from __future__ import annotations

import numpy as np
import shap


class SHAPExplainer:
    """Token-level SHAP attributions for the firewall classifier."""

    def __init__(self, model: object, max_evals: int = 500) -> None:
        self._model = model

        def _predict_proba(texts: list[str]) -> np.ndarray:
            results = model.predict(list(texts))
            return np.array([[v for v in r.values()] for r in results])

        self._explainer = shap.Explainer(
            _predict_proba,
            shap.maskers.Text(r"\W+"),
            max_evals=max_evals,
        )

    def explain(self, texts: list[str]) -> list[dict]:
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
    model: object,
    layer: int = -1,
    head: int = 0,
) -> None:
    """Render a matplotlib attention heatmap for a single input."""
    import matplotlib.pyplot as plt
    import torch

    tokenizer = model.tokenizer
    tokens = tokenizer.tokenize(text)

    encoding = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.model(**encoding, output_attentions=True)

    attn = outputs.attentions[layer][0, head].cpu().numpy()

    n = len(tokens)
    fig, ax = plt.subplots(figsize=(max(6, n // 2), max(6, n // 2)))
    im = ax.imshow(attn[:n, :n], cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_title(f"Attention layer {layer} head {head}")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()
