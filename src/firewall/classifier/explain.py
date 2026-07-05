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

    def __init__(self, model: FirewallClassifier, max_evals: int = _DEFAULT_MAX_EVALS) -> None:
        self._model = model

        def _predict_proba(texts: list[str]) -> np.ndarray[Any, np.dtype[Any]]:
            results = model.predict(list(texts))
            return np.array([list(r.values()) for r in results])

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
        return [
            {"tokens": shap_values.data[i], "shap_values": shap_values.values[i]}
            for i in range(len(texts))
        ]


def block_push_scores(
    shap_values: np.ndarray[Any, np.dtype[Any]], benign_index: int
) -> np.ndarray[Any, np.dtype[Any]]:
    """Project a ``[n_tokens, n_classes]`` SHAP matrix onto the benign↔threat decision axis.

    The firewall routes on benign-vs-threat, so the audit-relevant question is "which tokens
    made this look malicious?" — not "which tokens pushed toward whichever class happened to
    win." We answer it as the *negative* of each token's contribution to the ``benign`` class:
    raising ``P(benign)`` makes the prompt look safer, lowering it makes it look threatening.
    Returns a per-token score where **positive = pushes toward BLOCK** (threat), negative =
    toward benign — a single meaning that holds across every example.
    """
    return -np.asarray(shap_values)[:, benign_index]


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
    fig, ax = plt.subplots(
        figsize=(
            max(_MIN_FIGSIZE, n // _FIGSIZE_DIVISOR),
            max(_MIN_FIGSIZE, n // _FIGSIZE_DIVISOR),
        )
    )
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


def plot_shap_attribution(  # pragma: no cover
    texts: list[str],
    model: FirewallClassifier,
    output_path: str | Path = "shap_example.png",
    benign_label: str = "benign",
    max_evals: int = 200,
) -> Path:
    """Render token-level SHAP attributions coloured on the benign↔threat axis.

    Red = pushes the prompt toward BLOCK (threat); blue = pushes toward benign (see
    :func:`block_push_scores`). This is the security-audit view — the colour is consistent
    across examples, unlike colouring by the (varying) predicted class.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    explainer = SHAPExplainer(model, max_evals=max_evals)
    rows: list[dict[str, Any]] = []
    for text in texts:
        pred = model.predict([text])[0]
        top = max(pred, key=pred.__getitem__)
        benign_index = list(pred.keys()).index(benign_label)
        sv = explainer.explain([text])[0]
        rows.append(
            {
                "tokens": sv["tokens"],
                "scores": block_push_scores(sv["shap_values"], benign_index),
                "predicted": top,
                "confidence": pred[top],
            }
        )

    vmax = max((float(np.max(np.abs(r["scores"]))) for r in rows), default=1.0) or 1.0
    cmap = LinearSegmentedColormap.from_list("shap", ["#3b82f6", "#f8fafc", "#ef4444"])

    fig, axes = plt.subplots(len(rows), 1, figsize=(14, 1.2 * len(rows) + 1.2))
    axes = [axes] if len(rows) == 1 else list(axes)
    for ax, row in zip(axes, rows, strict=True):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(
            -0.01,
            0.5,
            f"{row['predicted']} ({row['confidence']:.0%})",
            fontsize=10,
            fontweight="bold",
            ha="right",
            va="center",
            transform=ax.transAxes,
            fontfamily="monospace",
        )
        x = 0.0
        for token, val in zip(row["tokens"], row["scores"] / vmax, strict=True):
            color = cmap((val + 1) / 2)  # [-1, 1] → blue…red
            txt = ax.text(
                x,
                0.5,
                f" {token if token.strip() else ' '} ",
                fontsize=11,
                va="center",
                ha="left",
                fontfamily="monospace",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": color,
                    "edgecolor": "none",
                    "alpha": 0.85,
                },
            )
            fig.canvas.draw()
            bb = txt.get_window_extent().transformed(ax.transData.inverted())
            x = bb.x1 + 0.002

    red = mpatches.Patch(color="#ef4444", alpha=0.85, label="Pushes toward BLOCK (threat)")
    blue = mpatches.Patch(color="#3b82f6", alpha=0.85, label="Pushes toward benign")
    fig.legend(
        handles=[red, blue],
        loc="lower center",
        ncol=2,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.05),
    )
    fig.suptitle(
        "SHAP Token Attribution (benign ↔ threat)", fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    out = Path(output_path)
    fig.savefig(out, dpi=_SAVE_DPI, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return out
