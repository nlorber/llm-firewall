# src/firewall/classifier/train.py
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from firewall.classifier.dataset import FirewallDataset, _load_jsonl

logger = logging.getLogger(__name__)


class WeightedTrainer(Trainer):
    """Trainer subclass that applies class weights to the cross-entropy loss."""

    def __init__(self, *, class_weights: torch.Tensor, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._class_weights = class_weights

    def compute_loss(  # type: ignore[override]  # HF Trainer signature uses Any broadly; our override is safe
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> Any:
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = torch.nn.functional.cross_entropy(
            logits, labels, weight=self._class_weights.to(logits.device),
        )
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """Compute accuracy + macro/weighted F1 for the HF Trainer callback."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":    float(accuracy_score(labels, preds)),
        "f1_macro":    float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }


def train(config_path: str | Path) -> None:
    """Fine-tune DeBERTa from configs/training.yaml."""
    config: dict[str, Any] = yaml.safe_load(Path(config_path).read_text())

    label_names: list[str] = config["label_names"]
    id2label = {i: lbl for i, lbl in enumerate(label_names)}
    label2id = {lbl: i for i, lbl in enumerate(label_names)}

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=config["num_labels"],
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    train_texts, train_labels = _load_jsonl(Path(config["train_path"]))
    val_texts, val_labels = _load_jsonl(Path(config["val_path"]))

    tokenizer_name = config["model_name"]
    train_ds = FirewallDataset(train_texts, train_labels, tokenizer_name, config["max_length"])
    val_ds   = FirewallDataset(val_texts, val_labels, tokenizer_name, config["max_length"])

    # Compute inverse-frequency class weights to handle imbalanced data
    weights = compute_class_weight("balanced", classes=np.arange(config["num_labels"]), y=np.array(train_labels))
    class_weights = torch.tensor(weights, dtype=torch.float32)

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        fp16=config.get("fp16", False),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=config["seed"],
        report_to="none",
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=config["early_stopping_patience"]
        )],
    )

    trainer.train()
    trainer.save_model(config["output_dir"])
    AutoTokenizer.from_pretrained(tokenizer_name).save_pretrained(config["output_dir"])  # type: ignore[no-untyped-call]
    logger.info("model saved to %s", config["output_dir"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa classifier")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
