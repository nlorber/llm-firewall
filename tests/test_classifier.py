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


class TestFirewallClassifier:
    def _make_mock_classifier(self):
        """Return a FirewallClassifier with all HF calls mocked out."""
        from firewall.classifier.model import FirewallClassifier

        with patch("firewall.classifier.model.AutoModelForSequenceClassification") as mock_m, \
             patch("firewall.classifier.model.AutoTokenizer") as mock_t:

            mock_model_inst = MagicMock()
            mock_model_inst.config.id2label = {
                0: "benign", 1: "injection", 2: "jailbreak",
                3: "exfiltration", 4: "escalation",
            }
            # Make .to() and .eval() return the same mock so clf.model == mock_model_inst
            mock_model_inst.to.return_value = mock_model_inst
            mock_model_inst.eval.return_value = mock_model_inst
            mock_logits = torch.tensor([[1.0, 0.1, 0.1, 0.1, 0.1]])
            mock_model_inst.return_value.logits = mock_logits
            mock_m.from_pretrained.return_value = mock_model_inst

            mock_tok_inst = MagicMock()
            mock_tok_inst.return_value = {
                "input_ids":      torch.zeros(1, 16, dtype=torch.long),
                "attention_mask": torch.ones(1, 16, dtype=torch.long),
            }
            mock_t.from_pretrained.return_value = mock_tok_inst

            clf = FirewallClassifier("dummy/model", num_labels=5)

        return clf, mock_model_inst, mock_tok_inst

    def test_predict_returns_one_dict_per_input(self) -> None:
        clf, mock_m, mock_t = self._make_mock_classifier()
        mock_m.return_value.logits = torch.zeros(2, 5)
        mock_t.return_value = {
            "input_ids":      torch.zeros(2, 16, dtype=torch.long),
            "attention_mask": torch.ones(2, 16, dtype=torch.long),
        }
        results = clf.predict(["hello", "ignore"])
        assert len(results) == 2

    def test_predict_output_has_all_label_keys(self) -> None:
        clf, _, _ = self._make_mock_classifier()
        results = clf.predict(["hello"])
        assert set(results[0].keys()) == {
            "benign", "injection", "jailbreak", "exfiltration", "escalation"
        }

    def test_predict_probabilities_sum_to_one(self) -> None:
        clf, _, _ = self._make_mock_classifier()
        results = clf.predict(["hello"])
        total = sum(results[0].values())
        assert abs(total - 1.0) < 1e-5


class TestComputeMetrics:
    def test_all_keys_present(self) -> None:
        import numpy as np

        from firewall.classifier.train import compute_metrics

        logits = np.array([[2.0, 0.1, 0.1, 0.1, 0.1], [0.1, 2.0, 0.1, 0.1, 0.1]])
        labels = np.array([0, 1])
        result = compute_metrics((logits, labels))
        assert "accuracy" in result
        assert "f1_macro" in result
        assert "f1_weighted" in result

    def test_perfect_predictions_give_f1_of_one(self) -> None:
        import numpy as np
        import pytest

        from firewall.classifier.train import compute_metrics

        logits = np.eye(5) * 10
        labels = np.arange(5)
        result = compute_metrics((logits, labels))
        assert result["accuracy"] == pytest.approx(1.0)
        assert result["f1_macro"] == pytest.approx(1.0)
