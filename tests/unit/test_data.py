# tests/test_data.py
from __future__ import annotations

import pytest

from data.prepare import LABEL_NAMES, deduplicate, harmonise_labels, stratified_split
from firewall.classifier.dataset import LABEL2ID, _load_jsonl


class TestHarmoniseLabels:
    def test_known_source_labels_are_mapped(self) -> None:
        records = [
            {"text": "Hello", "label": "0"},
            {"text": "Ignore prev", "label": "1"},
            {"text": "DAN prompt", "label": "jailbreak"},
        ]
        result = harmonise_labels(records)
        assert result[0]["label"] == "benign"
        assert result[1]["label"] == "injection"
        assert result[2]["label"] == "jailbreak"

    def test_all_output_labels_are_canonical(self) -> None:
        records = [{"text": "x", "label": str(i)} for i in range(2)]
        result = harmonise_labels(records)
        assert all(r["label"] in LABEL_NAMES for r in result)

    def test_unknown_label_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown label"):
            harmonise_labels([{"text": "x", "label": "UNKNOWN_999"}])


class TestDeduplicate:
    def test_removes_exact_duplicates(self) -> None:
        records = [
            {"text": "hello", "label": "benign"},
            {"text": "hello", "label": "benign"},
            {"text": "world", "label": "injection"},
        ]
        result = deduplicate(records)
        assert len(result) == 2

    def test_case_insensitive_dedup(self) -> None:
        records = [
            {"text": "HELLO", "label": "benign"},
            {"text": "hello", "label": "benign"},
        ]
        result = deduplicate(records)
        assert len(result) == 1
        assert result[0]["text"] == "HELLO"  # first occurrence preserved

    def test_preserves_first_occurrence(self) -> None:
        records = [
            {"text": "dup", "label": "benign"},
            {"text": "dup", "label": "injection"},
        ]
        result = deduplicate(records)
        assert result[0]["label"] == "benign"


class TestStratifiedSplit:
    def test_split_ratios_are_approximately_correct(self) -> None:
        records = [
            {"text": f"t{i}", "label": lbl} for i in range(20) for lbl in ["benign", "injection"]
        ]
        train, val, test_split = stratified_split(records, val_ratio=0.15, test_ratio=0.15)
        total = len(train) + len(val) + len(test_split)
        assert total == len(records)
        assert abs(len(val) / total - 0.15) < 0.05

    def test_all_classes_present_in_train(self) -> None:
        records = [
            {"text": f"t{i}", "label": lbl}
            for i in range(10)
            for lbl in ["benign", "injection", "jailbreak"]
        ]
        train, _, _ = stratified_split(records)
        labels_in_train = {r["label"] for r in train}
        assert labels_in_train == {"benign", "injection", "jailbreak"}

    def test_seed_produces_deterministic_splits(self) -> None:
        records = [
            {"text": f"t{i}", "label": lbl} for i in range(50) for lbl in ["benign", "injection"]
        ]
        split_a = stratified_split(records, seed=42)
        split_b = stratified_split(records, seed=42)
        assert [r["text"] for r in split_a[0]] == [r["text"] for r in split_b[0]]


class TestLoadJsonl:
    def test_loads_texts_and_labels_from_jsonl(self, tmp_path) -> None:
        jsonl = tmp_path / "data.jsonl"
        jsonl.write_text(
            '{"text": "hello", "label": "benign"}\n'
            '{"text": "ignore instructions", "label": "injection"}\n'
        )
        texts, labels = _load_jsonl(jsonl)
        assert texts == ["hello", "ignore instructions"]
        assert labels == [LABEL2ID["benign"], LABEL2ID["injection"]]

    def test_empty_file_returns_empty_lists(self, tmp_path) -> None:
        jsonl = tmp_path / "empty.jsonl"
        jsonl.write_text("")
        texts, labels = _load_jsonl(jsonl)
        assert texts == []
        assert labels == []
