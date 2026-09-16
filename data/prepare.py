# data/prepare.py
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
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

_SEED = 42
# Word-set overlap above which two prompts count as the same example. Paraphrases reuse most
# of their vocabulary, so this catches them while leaving distinct prompts of the same attack
# family alone.
_NEAR_DUP_JACCARD = 0.8
_MAX_SEEDS = 30
_SEED_CHARS = 300
# Variants requested per call, and the per-class size the corpus is levelled to — classes
# below it are topped up by paraphrase where the model will do it, classes above it are
# sampled down. Levelling from both sides is what keeps the benign class (which carries the
# roleplay hard negatives) from dominating the attack classes.
_AUGMENT_CHUNK = 50
_CLASS_TARGET = 250
# Long-document examples: how many benign prompts get concatenated into one document, and how
# many documents each split receives (attack-carrying and attack-free in equal number).
_DOC_PARTS = (4, 10)
_DOC_COUNTS = {"train": 160, "val": 40, "test": 40}


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


def _word_set(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"\w+", text.lower()))


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def deduplicate(records: list[dict]) -> list[dict]:
    """Drop exact and near-duplicate texts, keeping the first occurrence.

    Runs before the split: a prompt and its paraphrase on opposite sides of the split would
    measure the model on rows it effectively trained on.
    """
    # ponytail: O(n^2) word-set comparison, fine at this corpus's scale (~10^3 rows);
    # switch to MinHash/LSH if it grows past ~10^4.
    seen_exact: set[str] = set()
    kept: list[dict] = []
    kept_words: list[frozenset[str]] = []
    near = 0
    for r in records:
        key = r["text"].lower().strip()
        if key in seen_exact:
            continue
        words = _word_set(r["text"])
        if words and any(
            len(words & other) / len(words | other) >= _NEAR_DUP_JACCARD for other in kept_words
        ):
            near += 1
            continue
        seen_exact.add(key)
        kept.append(r)
        kept_words.append(words)
    before = len(records)
    print(
        f"[prepare] dedup: {before} → {len(kept)} records "
        f"({before - len(kept)} removed, {near} of them near-duplicates)"
    )
    return kept


def stratified_split(
    records: list[dict],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = _SEED,
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


def make_context_examples(records: list[dict], count: int, seed: int = _SEED) -> list[dict]:
    """Build long-document examples: benign prompts concatenated, half carrying an attack.

    Serving scores an over-length prompt as overlapping windows and keeps the most
    threatening one, so a window is typically a lot of benign text with at most one attack
    sentence inside it. Trained only on standalone prompts, the model reads such a window as
    benign and the attack is passed as CLEAN — below the gray band, so the judge never sees
    it either. These examples put that shape in the training distribution.

    ``count`` attack-carrying documents are produced *and* ``count`` attack-free ones: without
    the second half the model would simply learn that a long document is an attack.

    Built from ``records`` alone, so a document never mixes text across a split boundary.
    Returns the new records; the caller appends them.
    """
    rng = random.Random(seed)
    benign = [r["text"] for r in records if r["label"] == "benign"]
    attacks = [r for r in records if r["label"] != "benign"]
    if len(benign) < _DOC_PARTS[1] or not attacks:
        return []

    built: list[dict] = []
    for _ in range(count):
        parts = rng.sample(benign, rng.randint(*_DOC_PARTS))
        attack = rng.choice(attacks)
        parts.insert(rng.randint(0, len(parts)), attack["text"])
        built.append({"text": " ".join(parts), "label": attack["label"]})
    for _ in range(count):
        parts = rng.sample(benign, rng.randint(*_DOC_PARTS))
        built.append({"text": " ".join(parts), "label": "benign"})
    return built


def _paraphrase(
    client: anthropic.Anthropic,
    model: str,
    label: str,
    originals: list[str],
    n: int,
    rng: random.Random,
) -> list[str]:
    """Request ``n`` paraphrased variants for one class. Retries once, then raises.

    Raises:
        RuntimeError: if both attempts return something that is not a JSON array. A refusal
            reads as prose and lands here too: the assistant declines to mass-produce attack
            text, which is a legitimate answer and must surface rather than pass as success.
    """
    seeds = rng.sample(originals, min(_MAX_SEEDS, len(originals)))
    seed_texts = "\n".join(f"- {t[:_SEED_CHARS]}" for t in seeds)
    prompt = (
        "You are helping build a training dataset for an AI safety classifier "
        "that detects malicious prompts. This is for defensive security research.\n\n"
        f"Below are example prompts labelled '{label}':\n\n{seed_texts}\n\n"
        f"Generate {n} new paraphrased variants for the training set. "
        "Each must be a standalone user message (1-3 sentences) with diverse phrasing.\n\n"
        'Return ONLY a JSON array of strings: ["variant1", "variant2", ...]'
    )

    last = ""
    for _ in range(2):
        response = client.messages.create(
            model=model, max_tokens=4096, messages=[{"role": "user", "content": prompt}]
        )
        raw_text = response.content[0].text.strip()
        last = raw_text
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        start, end = raw_text.find("["), raw_text.rfind("]")
        if start == -1 or end == -1:
            continue
        try:
            variants = json.loads(raw_text[start : end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(variants, list):
            return [v for v in variants if isinstance(v, str) and v.strip()]
    raise RuntimeError(f"augmentation for '{label}' returned no usable JSON: {last[:200]!r}")


def augment(
    records: list[dict],
    target_per_class: int = _CLASS_TARGET,
    model: str = "claude-haiku-4-5-20251001",
    seed: int = _SEED,
) -> list[dict]:
    """Paraphrase examples in underrepresented classes towards ``target_per_class``.

    Takes the **training split only** — a paraphrase of a held-out row would leak it into
    training. Short classes are topped up in batches of ``_AUGMENT_CHUNK``; duplicates of text
    already in the corpus are discarded.

    A class the model declines to paraphrase is reported and left at its real size rather than
    abandoning the whole run: the assistant refuses to mass-produce injection and jailbreak
    text, so those classes are levelled by capping the larger ones instead (see
    :func:`cap_classes`). Every shortfall is printed, so a class is never quietly left short.
    """
    dist = Counter(r["label"] for r in records)
    client = anthropic.Anthropic()
    rng = random.Random(seed)
    augmented: list[dict] = list(records)
    seen = {_norm(r["text"]) for r in records}
    shortfalls: dict[str, int] = {}

    for label, count in sorted(dist.items()):
        needed = target_per_class - count
        if needed <= 0:
            continue
        originals = [r["text"] for r in records if r["label"] == label]
        added = 0
        while added < needed:
            batch = min(_AUGMENT_CHUNK, needed - added)
            try:
                variants = _paraphrase(client, model, label, originals, batch, rng)
            except RuntimeError as exc:
                print(f"[prepare] WARNING: {exc}")
                shortfalls[label] = count + added
                break
            fresh = 0
            for text in variants:
                key = _norm(text)
                if key in seen:
                    continue
                seen.add(key)
                augmented.append({"text": text, "label": label})
                added += 1
                fresh += 1
                if added >= needed:
                    break
            if fresh == 0:
                print(f"[prepare] WARNING: '{label}' batch returned only duplicates, stopping")
                shortfalls[label] = count + added
                break
        print(f"[prepare] augment '{label}': {count} → {count + added}")

    if shortfalls:
        print(f"[prepare] WARNING: below the {target_per_class} target: {shortfalls}")
    return augmented


def cap_classes(
    records: list[dict], target_per_class: int = _CLASS_TARGET, seed: int = _SEED
) -> list[dict]:
    """Sample classes larger than ``target_per_class`` down to it.

    The counterpart to :func:`augment`: where a class cannot be (or need not be) grown, the
    balance is made by shrinking the others. Without this, benign — which carries every
    roleplay hard negative — outnumbers each attack class and the model trades recall for
    precision.
    """
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)

    kept: list[dict] = []
    for label, rows in sorted(by_label.items()):
        if len(rows) > target_per_class:
            print(f"[prepare] cap '{label}': {len(rows)} → {target_per_class}")
            kept.extend(rng.sample(rows, target_per_class))
        else:
            kept.extend(rows)
    return kept


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[prepare] wrote {len(records)} records → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare merged dataset for training")
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

    # Split before augmenting: paraphrases are generated from training rows only, so none of
    # them can appear in val or test.
    train, val, test = stratified_split(records)
    print(f"[prepare] split: train={len(train)} val={len(val)} test={len(test)}")

    if not args.skip_augment:
        train = augment(train)
    else:
        print("[prepare] skipping augmentation (--skip-augment)")
    train = cap_classes(train)

    # Long-document examples come last so the capping step cannot discard them. Each split
    # builds its own from its own rows.
    for name, split in (("train", train), ("val", val), ("test", test)):
        extra = make_context_examples(split, _DOC_COUNTS[name])
        split.extend(extra)
        print(f"[prepare] {name}: +{len(extra)} long-document examples (half attack-carrying)")

    print("[prepare] train class distribution:", dict(Counter(r["label"] for r in train)))
    print("[prepare] val class distribution:", dict(Counter(r["label"] for r in val)))
    print("[prepare] test class distribution:", dict(Counter(r["label"] for r in test)))

    _write_jsonl(train, args.output_dir / "train.jsonl")
    _write_jsonl(val, args.output_dir / "val.jsonl")
    _write_jsonl(test, args.output_dir / "test.jsonl")


if __name__ == "__main__":
    main()
