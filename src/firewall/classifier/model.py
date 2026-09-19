from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast

from firewall.classifier.dataset import DEFAULT_MAX_LENGTH, NUM_LABELS

# Tokens shared by consecutive windows of an over-length prompt, so a short injection that
# straddles a window boundary still appears whole in one window.
_WINDOW_STRIDE = 128
_OVERFLOW_KEY = "overflow_to_sample_mapping"


def threat_score(scores: dict[str, float]) -> float:
    """Max probability across non-benign classes — the score the orchestrator routes on."""
    return max((v for k, v in scores.items() if k != "benign"), default=0.0)


class FirewallClassifier:
    """Fine-tuned DeBERTa-v3-base sequence classifier for prompt threat detection."""

    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int = NUM_LABELS,
        device: str | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path, num_labels=num_labels
            )
            .to(self.device)
            .eval()
        )
        # The tokenizer exactly as serialized with the checkpoint. AutoTokenizer rebuilds
        # DeBERTa-v2's pipeline from spm.model without the saved NFKC normalizer, so fullwidth,
        # ligature and homoglyph text would tokenize differently from training.
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name_or_path)
        self.id2label: dict[int, str] = dict(self.model.config.id2label)

    def predict(self, texts: list[str]) -> list[dict[str, float]]:
        """Return per-class probabilities for a batch of prompt strings.

        A prompt longer than ``max_length`` tokens is scored as overlapping windows and
        represented by its most threatening window. Plain truncation would drop the tail
        unscored, letting an injection hide behind a long benign preamble.
        """
        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            stride=_WINDOW_STRIDE,
            return_overflowing_tokens=True,
            return_tensors="pt",
        )
        owners = encoding[_OVERFLOW_KEY].tolist()  # window index -> index of its text
        inputs = {k: v.to(self.device) for k, v in encoding.items() if k != _OVERFLOW_KEY}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
        best: dict[int, dict[str, float]] = {}
        for owner, prob in zip(owners, probs, strict=True):
            scores = {self.id2label[i]: float(p) for i, p in enumerate(prob)}
            if owner not in best or threat_score(scores) > threat_score(best[owner]):
                best[owner] = scores
        return [best[i] for i in range(len(texts))]

    def save(self, output_dir: str | Path) -> None:
        """Save model weights + tokenizer to output_dir."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out)
        self.tokenizer.save_pretrained(out)


def load_classifier(
    checkpoint_path: str | Path,
    max_length: int = 512,
) -> FirewallClassifier:
    """Load a fine-tuned FirewallClassifier from a checkpoint directory."""
    return FirewallClassifier(str(checkpoint_path), max_length=max_length)
