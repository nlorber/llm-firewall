# src/firewall/classifier/dataset.py
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from pathlib import Path

LABEL_NAMES: list[str] = ["benign", "injection", "jailbreak", "exfiltration", "escalation"]
NUM_LABELS: int = len(LABEL_NAMES)
LABEL2ID: dict[str, int] = {lbl: i for i, lbl in enumerate(LABEL_NAMES)}
DEFAULT_MAX_LENGTH: int = 512


class FirewallDataset(Dataset[dict[str, torch.Tensor]]):
    """Tokenises raw texts up-front and stores tensors in memory.

    With ~1,300 examples the encoded data fits comfortably in RAM.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int] | None = None,
        tokenizer_name: str = "microsoft/deberta-v3-base",
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)  # type: ignore[no-untyped-call]
        encoding = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self._input_ids = encoding["input_ids"]
        self._attention_mask = encoding["attention_mask"]
        self._token_type_ids = encoding.get("token_type_ids")
        self._labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self) -> int:
        return int(self._input_ids.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item: dict[str, torch.Tensor] = {
            "input_ids": self._input_ids[idx],
            "attention_mask": self._attention_mask[idx],
        }
        if self._token_type_ids is not None:
            item["token_type_ids"] = self._token_type_ids[idx]
        if self._labels is not None:
            item["labels"] = self._labels[idx]
        return item


def _load_jsonl(path: Path) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            r = json.loads(line)
            raw_label = r["label"]
            if raw_label not in LABEL2ID:
                msg = f"Unknown label {raw_label!r} at line {lineno} in {path}"
                raise ValueError(msg)
            texts.append(r["text"])
            labels.append(LABEL2ID[raw_label])
    return texts, labels
