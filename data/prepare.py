"""Prepare the merged dataset for training: clean, deduplicate, split, augment.

Usage::

    python data/prepare.py --input-dir data/raw --output-dir data/processed

Pipeline:
    1. Load and merge all raw JSONL files from ``data/raw/``
    2. Harmonise heterogeneous label strings to the canonical 5-class taxonomy
    3. Deduplicate on normalised text (lowercased, stripped)
    4. Stratified train / val / test split (70 / 15 / 15, seed 42)
    5. Optional: LLM-based paraphrase augmentation for underrepresented classes
    6. Write ``train.jsonl``, ``val.jsonl``, ``test.jsonl`` to ``data/processed/``
"""
from __future__ import annotations

from pathlib import Path

LABEL_NAMES: list[str] = ["benign", "injection", "jailbreak", "exfiltration", "escalation"]


def load_raw(input_dir: Path) -> list[dict]:
    """Load and merge all JSONL files from *input_dir*.

    Args:
        input_dir: Directory containing raw ``.jsonl`` files.

    Returns:
        List of dicts with at least ``text`` and ``label`` keys.
    """
    raise NotImplementedError


def harmonise_labels(records: list[dict]) -> list[dict]:
    """Map source-specific label strings to the canonical 5-class taxonomy.

    Args:
        records: Raw records with heterogeneous label strings.

    Returns:
        Records with ``label`` values restricted to :data:`LABEL_NAMES`.
    """
    raise NotImplementedError


def deduplicate(records: list[dict]) -> list[dict]:
    """Remove duplicate records based on normalised (lowercased, stripped) text.

    Args:
        records: List of labelled records.

    Returns:
        Deduplicated list preserving the first occurrence.
    """
    raise NotImplementedError


def stratified_split(
    records: list[dict],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records into train / val / test while preserving class distribution.

    Args:
        records: Full labelled dataset.
        val_ratio: Fraction of data for validation set.
        test_ratio: Fraction of data for test set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of ``(train, val, test)`` record lists.
    """
    raise NotImplementedError


def augment(
    records: list[dict],
    target_class: str,
    n_variants: int = 3,
) -> list[dict]:
    """Generate paraphrase variants for *target_class* via LLM augmentation.

    Uses the Claude API to produce semantically equivalent but lexically diverse
    examples. Caller is responsible for merging returned records into the dataset.

    Args:
        records: Existing labelled records for *target_class*.
        target_class: Label string of the underrepresented class.
        n_variants: Number of paraphrases to generate per source example.

    Returns:
        New augmented records (not merged — caller decides inclusion strategy).
    """
    raise NotImplementedError


def main() -> None:
    """CLI entry point: parse arguments and run the full preparation pipeline."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
