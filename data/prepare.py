# data/prepare.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anthropic
from sklearn.model_selection import StratifiedShuffleSplit

# Canonical source: firewall.classifier.dataset.LABEL_NAMES (duplicated here
# because data/ scripts are standalone and don't depend on the firewall package).
LABEL_NAMES: list[str] = ["benign", "injection", "jailbreak", "exfiltration", "escalation"]

# Maps raw label values (as strings) from each source to canonical LABEL_NAMES
_LABEL_MAP: dict[str, str] = {
    "0": "benign",
    "1": "injection",
    "benign": "benign",
    "injection": "injection",
    "jailbreak": "jailbreak",
    "exfiltration": "exfiltration",
    "escalation": "escalation",
}


def load_raw(input_dir: Path) -> list[dict]:
    """Load and merge all JSONL files from input_dir."""
    records: list[dict] = []
    for path in sorted(input_dir.glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    print(f"[prepare] loaded {len(records)} raw records from {input_dir}")
    return records


def harmonise_labels(records: list[dict]) -> list[dict]:
    """Map source-specific label strings to the canonical 5-class taxonomy."""
    result = []
    for r in records:
        raw = str(r["label"]).strip().lower()
        if raw not in _LABEL_MAP:
            raise ValueError(f"unknown label '{r['label']}' in record: {r['text'][:60]!r}")
        result.append({**r, "label": _LABEL_MAP[raw]})
    return result


def deduplicate(records: list[dict]) -> list[dict]:
    """Remove duplicates by normalised (lowercased, stripped) text, keep first."""
    seen: set[str] = set()
    result: list[dict] = []
    for r in records:
        key = r["text"].lower().strip()
        if key not in seen:
            seen.add(key)
            result.append(r)
    before = len(records)
    print(f"[prepare] dedup: {before} → {len(result)} records ({before - len(result)} removed)")
    return result


def stratified_split(
    records: list[dict],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratified train/val/test split preserving class distribution."""
    texts = [r["text"] for r in records]
    labels = [r["label"] for r in records]

    # First split: carve off test set
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(texts, labels))

    trainval_records = [records[i] for i in trainval_idx]
    test_records = [records[i] for i in test_idx]

    # Second split: carve val out of trainval
    adjusted_val = val_ratio / (1 - test_ratio)
    tv_texts = [r["text"] for r in trainval_records]
    tv_labels = [r["label"] for r in trainval_records]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=adjusted_val, random_state=seed)
    train_idx, val_idx = next(sss2.split(tv_texts, tv_labels))

    train = [trainval_records[i] for i in train_idx]
    val = [trainval_records[i] for i in val_idx]
    return train, val, test_records


def augment(
    records: list[dict],
    target_per_class: int | None = None,
    model: str = "claude-haiku-4-5-20251001",
) -> list[dict]:
    """Paraphrase examples in underrepresented classes to balance the dataset.

    For each class below ``target_per_class``, generates paraphrased variants
    via Claude until the class reaches the target count. If ``target_per_class``
    is None, uses the count of the largest class.
    """
    from collections import Counter

    dist = Counter(r["label"] for r in records)
    if target_per_class is None:
        target_per_class = max(dist.values())

    client = anthropic.Anthropic()
    augmented: list[dict] = list(records)

    for label, count in dist.items():
        needed = target_per_class - count
        if needed <= 0:
            continue

        # Sample existing examples as paraphrase seeds
        originals = [r["text"] for r in records if r["label"] == label]
        seed_texts = "\n".join(f"- {t}" for t in originals[:30])

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": (
                    "You are helping build a training dataset for an AI safety classifier "
                    "that detects malicious prompts. This is for defensive security research.\n\n"
                    f"Below are example prompts labelled '{label}':\n\n{seed_texts}\n\n"
                    f"Generate {needed} new paraphrased variants for the training set. "
                    "Each must be a standalone user message (1-3 sentences) with diverse phrasing.\n\n"
                    'Return ONLY a JSON array of strings: ["variant1", "variant2", ...]'
                ),
            }],
        )
        raw_text = response.content[0].text.strip()
        # Strip markdown code fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        # Extract JSON array if surrounded by other text
        start = raw_text.find("[")
        end = raw_text.rfind("]")
        if start != -1 and end != -1:
            raw_text = raw_text[start:end + 1]
        if not raw_text:
            print(f"[prepare] WARNING: empty response for '{label}', skipping")
            continue
        try:
            variants: list[str] = json.loads(raw_text)
        except json.JSONDecodeError:
            print(f"[prepare] WARNING: unparseable response for '{label}': {raw_text[:200]!r}, skipping")
            continue
        for text in variants[:needed]:
            augmented.append({"text": text, "label": label})

        print(f"[prepare] augment '{label}': {count} → {count + min(len(variants), needed)}")

    return augmented


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[prepare] wrote {len(records)} records → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare merged dataset for training"
    )
    parser.add_argument("--input-dir", default="data/raw", type=Path)
    parser.add_argument("--output-dir", default="data/processed", type=Path)
    parser.add_argument(
        "--skip-augment",
        action="store_true",
        help="Skip LLM-based paraphrasing (no ANTHROPIC_API_KEY required)",
    )
    args = parser.parse_args()

    records = load_raw(args.input_dir)
    records = harmonise_labels(records)
    records = deduplicate(records)

    if not args.skip_augment:
        records = augment(records)
    else:
        print("[prepare] skipping augmentation (--skip-augment)")

    from collections import Counter

    dist = Counter(r["label"] for r in records)
    print("[prepare] class distribution:", dict(dist))

    train, val, test = stratified_split(records)

    _write_jsonl(train, args.output_dir / "train.jsonl")
    _write_jsonl(val, args.output_dir / "val.jsonl")
    _write_jsonl(test, args.output_dir / "test.jsonl")


if __name__ == "__main__":
    main()
