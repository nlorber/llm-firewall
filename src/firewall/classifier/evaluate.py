from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from firewall.classifier.dataset import LABEL2ID
from firewall.classifier.model import load_classifier


def evaluate(
    model_path: str | Path,
    test_path: str | Path,
    label_names: list[str],
) -> dict[str, Any]:
    """Run inference on the test split and return full evaluation metrics."""
    clf = load_classifier(model_path)

    texts, y_true = [], []
    with Path(test_path).open() as f:
        for line in f:
            r = json.loads(line)
            texts.append(r["text"])
            y_true.append(LABEL2ID[r["label"]])

    # Batch inference to avoid OOM on large test sets
    batch_size = 32
    all_pred_labels: list[str] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        results = clf.predict(batch)
        for r in results:
            all_pred_labels.append(max(r, key=r.__getitem__))

    y_pred = [LABEL2ID[p] for p in all_pred_labels]

    all_label_ids = list(range(len(label_names)))
    return {
        "accuracy":              float(accuracy_score(y_true, y_pred)),
        "f1_macro":              float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=all_label_ids)),
        "f1_weighted":           float(f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=all_label_ids)),
        "confusion_matrix":      confusion_matrix(y_true, y_pred, labels=all_label_ids),
        "classification_report": classification_report(
            y_true, y_pred, labels=all_label_ids, target_names=label_names, zero_division=0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned classifier")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    results = evaluate(
        model_path=config["output_dir"],
        test_path=config["test_path"],
        label_names=config["label_names"],
    )
    print(f"Accuracy:    {results['accuracy']:.4f}")
    print(f"F1 macro:    {results['f1_macro']:.4f}")
    print(f"F1 weighted: {results['f1_weighted']:.4f}")
    print("\nClassification report:\n", results["classification_report"])


if __name__ == "__main__":
    main()
