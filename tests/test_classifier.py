# tests/test_classifier.py
from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch


class TestFirewallDataset:
    def _mock_tokenizer_output(self, n: int, seq_len: int = 16) -> dict:
        return {
            "input_ids":      torch.zeros(n, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(n, seq_len, dtype=torch.long),
        }

    def test_dataset_length_matches_texts(self) -> None:
        from firewall.classifier.dataset import FirewallDataset

        texts = ["hello", "world", "test"]
        labels = [0, 1, 2]
        with patch("firewall.classifier.dataset.AutoTokenizer") as mock_tok_cls:
            mock_tok = MagicMock()
            mock_tok.return_value = self._mock_tokenizer_output(3)
            mock_tok_cls.from_pretrained.return_value = mock_tok
            ds = FirewallDataset(texts, labels, tokenizer_name="dummy/tokenizer")

        assert len(ds) == 3

    def test_item_contains_input_ids_and_attention_mask(self) -> None:
        from firewall.classifier.dataset import FirewallDataset

        texts = ["hello", "world"]
        with patch("firewall.classifier.dataset.AutoTokenizer") as mock_tok_cls:
            mock_tok = MagicMock()
            mock_tok.return_value = self._mock_tokenizer_output(2)
            mock_tok_cls.from_pretrained.return_value = mock_tok
            ds = FirewallDataset(texts, tokenizer_name="dummy/tokenizer")

        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert isinstance(item["input_ids"], torch.Tensor)

    def test_item_contains_labels_when_provided(self) -> None:
        from firewall.classifier.dataset import FirewallDataset

        texts = ["a", "b"]
        labels = [0, 3]
        with patch("firewall.classifier.dataset.AutoTokenizer") as mock_tok_cls:
            mock_tok = MagicMock()
            mock_tok.return_value = self._mock_tokenizer_output(2)
            mock_tok_cls.from_pretrained.return_value = mock_tok
            ds = FirewallDataset(texts, labels, tokenizer_name="dummy/tokenizer")

        assert ds[1]["labels"].item() == 3

    def test_no_labels_key_when_labels_not_provided(self) -> None:
        from firewall.classifier.dataset import FirewallDataset

        with patch("firewall.classifier.dataset.AutoTokenizer") as mock_tok_cls:
            mock_tok = MagicMock()
            mock_tok.return_value = self._mock_tokenizer_output(1)
            mock_tok_cls.from_pretrained.return_value = mock_tok
            ds = FirewallDataset(["x"], tokenizer_name="dummy/tokenizer")

        assert "labels" not in ds[0]
