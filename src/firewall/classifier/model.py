from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from firewall.classifier.dataset import NUM_LABELS


class FirewallClassifier:
    """Fine-tuned DeBERTa-v3-base sequence classifier for prompt threat detection."""

    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int = NUM_LABELS,
        device: str | None = None,
        max_length: int = 512,
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)  # type: ignore[no-untyped-call]
        self.id2label: dict[int, str] = dict(self.model.config.id2label)

    def predict(self, texts: list[str]) -> list[dict[str, float]]:
        """Return per-class probabilities for a batch of prompt strings."""
        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        with torch.no_grad():
            logits = self.model(**encoding).logits
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
        return [
            {self.id2label[i]: float(p) for i, p in enumerate(prob)}
            for prob in probs
        ]

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
