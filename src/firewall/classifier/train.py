# src/firewall/classifier/train.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from firewall.classifier.dataset import LABEL2ID, FirewallDataset


def compute_metrics(eval_pred: tuple) -> dict[str, float]:
    """Compute accuracy + macro/weighted F1 for the HF Trainer callback."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":    float(accuracy_score(labels, preds)),
        "f1_macro":    float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }


def _load_jsonl(path: Path) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            texts.append(r["text"])
            labels.append(LABEL2ID[r["label"]])
    return texts, labels


def train(config_path: str | Path) -> None:
    """Fine-tune DeBERTa from configs/training.yaml."""
    config = yaml.safe_load(Path(config_path).read_text())

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

    trainer = Trainer(
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
    AutoTokenizer.from_pretrained(tokenizer_name).save_pretrained(config["output_dir"])
    print(f"[train] model saved to {config['output_dir']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa classifier")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
