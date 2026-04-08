# src/firewall/classifier/dataset.py
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

LABEL_NAMES: list[str] = ["benign", "injection", "jailbreak", "exfiltration", "escalation"]
LABEL2ID: dict[str, int] = {lbl: i for i, lbl in enumerate(LABEL_NAMES)}


class FirewallDataset(Dataset[dict[str, torch.Tensor]]):
    """Tokenises raw texts up-front and stores tensors in memory.

    With ~5k examples and max_length=512, the encoded data fits comfortably in RAM.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int] | None = None,
        tokenizer_name: str = "microsoft/deberta-v3-base",
        max_length: int = 512,
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
        self._labels = (
            torch.tensor(labels, dtype=torch.long) if labels is not None else None
        )

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
        for line in f:
            r = json.loads(line)
            texts.append(r["text"])
            labels.append(LABEL2ID[r["label"]])
    return texts, labels


def create_dataloaders(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    tokenizer_name: str,
    batch_size: int,
    max_length: int = 512,
) -> tuple[DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]]]:
    """Build DataLoaders from processed JSONL splits."""

    def _make(path: Path, shuffle: bool) -> DataLoader[dict[str, torch.Tensor]]:
        texts, labels = _load_jsonl(path)
        ds = FirewallDataset(texts, labels, tokenizer_name, max_length)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    return (
        _make(train_path, shuffle=True),
        _make(val_path, shuffle=False),
        _make(test_path, shuffle=False),
    )
