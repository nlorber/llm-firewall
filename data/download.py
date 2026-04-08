"""Download raw datasets from HuggingFace Hub.

Usage::

    python data/download.py --output-dir data/raw

Datasets downloaded:
    - ``deepset/prompt-injections``: ~600 labelled injection + benign examples
    - ``JailbreakBench/JBB-Behaviors``: structured jailbreak prompts

Each dataset is saved as a ``.jsonl`` file under *output_dir*.
"""
from __future__ import annotations

from pathlib import Path


def download_prompt_injections(output_dir: Path) -> None:
    """Download ``deepset/prompt-injections`` and save as ``prompt_injections.jsonl``.

    Args:
        output_dir: Directory to write the raw JSONL file.
    """
    raise NotImplementedError


def download_jailbreak_bench(output_dir: Path) -> None:
    """Download ``JailbreakBench/JBB-Behaviors`` and save as ``jailbreak_bench.jsonl``.

    Args:
        output_dir: Directory to write the raw JSONL file.
    """
    raise NotImplementedError


def main() -> None:
    """CLI entry point: parse ``--output-dir`` and run all downloads."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
