from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from firewall.classifier.dataset import LABEL2ID, _load_jsonl
from firewall.classifier.model import load_classifier

logger = logging.getLogger(__name__)

_DEFAULT_EVAL_BATCH_SIZE = 32
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def evaluate(
    model_path: str | Path,
    test_path: str | Path,
    label_names: list[str],
    batch_size: int = _DEFAULT_EVAL_BATCH_SIZE,
) -> dict[str, Any]:
    """Run inference on the test split and return full evaluation metrics."""
    clf = load_classifier(model_path)

    texts, y_true = _load_jsonl(Path(test_path))
    all_pred_labels: list[str] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        results = clf.predict(batch)
        for r in results:
            all_pred_labels.append(max(r, key=r.__getitem__))

    y_pred = [LABEL2ID[p] for p in all_pred_labels]

    all_label_ids = list(range(len(label_names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0, labels=all_label_ids)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=all_label_ids)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=all_label_ids),
        "classification_report": classification_report(
            y_true, y_pred, labels=all_label_ids, target_names=label_names, zero_division=0
        ),
    }


def threat_score(scores: dict[str, float]) -> float:
    """Max probability across non-benign classes — mirrors the orchestrator's routing."""
    return max((v for k, v in scores.items() if k != "benign"), default=0.0)


def compute_robustness_metrics(
    predictions: list[dict[str, float]],
    true_labels: list[str],
    attack_types: list[str],
    clean_threshold: float,
    block_threshold: float,
) -> dict[str, Any]:
    """Aggregate out-of-distribution detection metrics for an all-threat eval set.

    The set contains only attack prompts (no benign examples), so this measures
    *recall* on novel attacks — the rate at which the firewall refuses to wave a
    prompt straight through as CLEAN — not precision or false-positive rate.

    - ``detection_rate``: fraction with ``threat_score >= clean_threshold`` (routed
      to the judge or blocked rather than passed as CLEAN).
    - ``block_rate``: fraction with ``threat_score >= block_threshold`` (hard-blocked
      by the classifier alone, no judge needed).
    - ``exact_class_accuracy``: fraction whose top predicted label equals the true
      attack class — much stricter than detection.
    """
    n = len(predictions)
    if n == 0:
        raise ValueError("no predictions to evaluate")

    detected = blocked = exact = 0
    per_type: dict[str, dict[str, int]] = {}
    for scores, true_label, atk in zip(predictions, true_labels, attack_types, strict=True):
        ts = threat_score(scores)
        top_label = max(scores, key=scores.__getitem__)
        is_detected = ts >= clean_threshold
        is_blocked = ts >= block_threshold
        is_exact = top_label == true_label

        detected += int(is_detected)
        blocked += int(is_blocked)
        exact += int(is_exact)

        bucket = per_type.setdefault(atk, {"n": 0, "detected": 0, "blocked": 0, "exact": 0})
        bucket["n"] += 1
        bucket["detected"] += int(is_detected)
        bucket["blocked"] += int(is_blocked)
        bucket["exact"] += int(is_exact)

    return {
        "n": n,
        "detection_rate": detected / n,
        "block_rate": blocked / n,
        "exact_class_accuracy": exact / n,
        "per_attack_type": per_type,
    }


def evaluate_robustness(
    model_path: str | Path,
    data_path: str | Path,
    clean_threshold: float,
    block_threshold: float,
    batch_size: int = _DEFAULT_EVAL_BATCH_SIZE,
) -> dict[str, Any]:
    """Evaluate the classifier on a held-out, all-threat OOD set.

    Each line is ``{"text", "label", "attack_type"}``. Benign labels are rejected:
    this set exists to measure detection recall on novel attacks, not FPR.
    """
    clf = load_classifier(model_path)

    texts: list[str] = []
    true_labels: list[str] = []
    attack_types: list[str] = []
    path = Path(data_path)
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            r = json.loads(stripped)
            label = r["label"]
            if label not in LABEL2ID:
                raise ValueError(f"Unknown label {label!r} at line {lineno} in {path}")
            if label == "benign":
                msg = f"robustness set must contain only threats; got benign at line {lineno}"
                raise ValueError(msg)
            texts.append(r["text"])
            true_labels.append(label)
            attack_types.append(r.get("attack_type", "unknown"))

    predictions: list[dict[str, float]] = []
    for i in range(0, len(texts), batch_size):
        predictions.extend(clf.predict(texts[i : i + batch_size]))

    return compute_robustness_metrics(
        predictions, true_labels, attack_types, clean_threshold, block_threshold
    )


def _write_metrics(results: dict[str, Any], path: Path) -> None:
    """Persist the JSON-serialisable headline metrics so the README numbers have a
    committed source of truth (mirrors transaction-classifier/reports/metrics.json)."""
    cm = results["confusion_matrix"]
    payload = {
        "accuracy": results["accuracy"],
        "f1_macro": results["f1_macro"],
        "f1_weighted": results["f1_weighted"],
        "confusion_matrix": cm.tolist() if hasattr(cm, "tolist") else cm,
        "classification_report": results["classification_report"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, stream=sys.stdout, force=True)
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned classifier")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("reports/metrics.json"),
        help="Where to write the JSON metrics artifact.",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    results = evaluate(
        model_path=config["output_dir"],
        test_path=config["test_path"],
        label_names=config["label_names"],
    )
    logger.info("Accuracy:    %.4f", results["accuracy"])
    logger.info("F1 macro:    %.4f", results["f1_macro"])
    logger.info("F1 weighted: %.4f", results["f1_weighted"])
    logger.info("Classification report:\n%s", results["classification_report"])

    _write_metrics(results, args.metrics_path)
    logger.info("Wrote metrics to %s", args.metrics_path)


def robustness_main() -> None:
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, stream=sys.stdout, force=True)
    parser = argparse.ArgumentParser(
        description="Evaluate OOD detection recall on a held-out attack set"
    )
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--clean-threshold", type=float, default=0.3)
    parser.add_argument("--block-threshold", type=float, default=0.8)
    args = parser.parse_args()

    results = evaluate_robustness(
        model_path=args.model_path,
        data_path=args.data_path,
        clean_threshold=args.clean_threshold,
        block_threshold=args.block_threshold,
    )
    logger.info("OOD set size:            %d", results["n"])
    logger.info("Detection rate (recall): %.4f", results["detection_rate"])
    logger.info("Block rate:              %.4f", results["block_rate"])
    logger.info("Exact-class accuracy:    %.4f", results["exact_class_accuracy"])
    logger.info("Per attack type:")
    for atk, b in sorted(results["per_attack_type"].items()):
        logger.info(
            "  %-22s n=%d detected=%d blocked=%d exact=%d",
            atk,
            b["n"],
            b["detected"],
            b["blocked"],
            b["exact"],
        )


if __name__ == "__main__":
    main()
