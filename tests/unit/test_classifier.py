# tests/test_classifier.py
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch


class TestFirewallDataset:
    def _mock_tokenizer_output(self, n: int, seq_len: int = 16) -> dict:
        return {
            "input_ids": torch.zeros(n, seq_len, dtype=torch.long),
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

    def test_item_includes_token_type_ids_when_present(self) -> None:
        from firewall.classifier.dataset import FirewallDataset

        texts = ["hello"]
        labels = [0]
        with patch("firewall.classifier.dataset.AutoTokenizer") as mock_tok_cls:
            mock_tok = MagicMock()
            mock_tok.return_value = {
                "input_ids": torch.zeros(1, 16, dtype=torch.long),
                "attention_mask": torch.ones(1, 16, dtype=torch.long),
                "token_type_ids": torch.zeros(1, 16, dtype=torch.long),
            }
            mock_tok_cls.from_pretrained.return_value = mock_tok
            ds = FirewallDataset(texts, labels, tokenizer_name="dummy/tokenizer")

        assert "token_type_ids" in ds[0]


class TestFirewallClassifier:
    def _make_mock_classifier(self):
        """Return a FirewallClassifier with all HF calls mocked out."""
        from firewall.classifier.model import FirewallClassifier

        with (
            patch("firewall.classifier.model.AutoModelForSequenceClassification") as mock_m,
            patch("firewall.classifier.model.AutoTokenizer") as mock_t,
        ):
            mock_model_inst = MagicMock()
            mock_model_inst.config.id2label = {
                0: "benign",
                1: "injection",
                2: "jailbreak",
                3: "exfiltration",
                4: "escalation",
            }
            # Make .to() and .eval() return the same mock so clf.model == mock_model_inst
            mock_model_inst.to.return_value = mock_model_inst
            mock_model_inst.eval.return_value = mock_model_inst
            mock_logits = torch.tensor([[1.0, 0.1, 0.1, 0.1, 0.1]])
            mock_model_inst.return_value.logits = mock_logits
            mock_m.from_pretrained.return_value = mock_model_inst

            mock_tok_inst = MagicMock()
            mock_tok_inst.return_value = {
                "input_ids": torch.zeros(1, 16, dtype=torch.long),
                "attention_mask": torch.ones(1, 16, dtype=torch.long),
            }
            mock_t.from_pretrained.return_value = mock_tok_inst

            clf = FirewallClassifier("dummy/model", num_labels=5)

        return clf, mock_model_inst, mock_tok_inst

    def test_predict_returns_one_dict_per_input(self) -> None:
        clf, mock_m, mock_t = self._make_mock_classifier()
        mock_m.return_value.logits = torch.zeros(2, 5)
        mock_t.return_value = {
            "input_ids": torch.zeros(2, 16, dtype=torch.long),
            "attention_mask": torch.ones(2, 16, dtype=torch.long),
        }
        results = clf.predict(["hello", "ignore"])
        assert len(results) == 2

    def test_predict_output_has_all_label_keys(self) -> None:
        clf, _, _ = self._make_mock_classifier()
        results = clf.predict(["hello"])
        assert set(results[0].keys()) == {
            "benign",
            "injection",
            "jailbreak",
            "exfiltration",
            "escalation",
        }

    def test_predict_probabilities_sum_to_one(self) -> None:
        clf, _, _ = self._make_mock_classifier()
        results = clf.predict(["hello"])
        total = sum(results[0].values())
        assert abs(total - 1.0) < 1e-5

    def test_save_creates_directory_and_persists(self, tmp_path) -> None:
        clf, mock_model, mock_tok = self._make_mock_classifier()
        save_dir = tmp_path / "saved_model"
        clf.save(save_dir)
        assert save_dir.exists()
        mock_model.save_pretrained.assert_called_once_with(save_dir)
        mock_tok.save_pretrained.assert_called_once_with(save_dir)


class TestWeightedTrainer:
    @staticmethod
    def _make_trainer(class_weights: torch.Tensor) -> Any:
        """Create a WeightedTrainer without invoking the full HF Trainer __init__."""
        from firewall.classifier.train import WeightedTrainer

        trainer = object.__new__(WeightedTrainer)
        trainer._class_weights = class_weights
        return trainer

    def test_compute_loss_applies_class_weights(self) -> None:
        class_weights = torch.tensor([1.0, 2.0, 3.0])
        logits = torch.tensor([[2.0, 0.5, 0.1]], dtype=torch.float32)
        labels = torch.tensor([1])

        mock_model = MagicMock()
        mock_model.return_value.logits = logits

        trainer = self._make_trainer(class_weights)
        inputs: dict[str, Any] = {
            "input_ids": torch.zeros(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "labels": labels,
        }
        loss = trainer.compute_loss(mock_model, inputs)
        expected = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights)
        assert loss.item() == pytest.approx(expected.item())

    def test_compute_loss_returns_tuple_when_requested(self) -> None:
        class_weights = torch.tensor([1.0, 1.0, 1.0])
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
        mock_model.return_value = mock_outputs

        trainer = self._make_trainer(class_weights)
        inputs: dict[str, Any] = {
            "input_ids": torch.zeros(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "labels": torch.tensor([0]),
        }
        result = trainer.compute_loss(mock_model, inputs, return_outputs=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[1] is mock_outputs


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


class TestEvaluate:
    def test_evaluate_returns_expected_keys(self, tmp_path) -> None:
        from unittest.mock import patch

        from firewall.classifier.evaluate import evaluate

        mock_clf = MagicMock()
        mock_clf.predict.return_value = [
            {
                "benign": 0.8,
                "injection": 0.1,
                "jailbreak": 0.05,
                "exfiltration": 0.03,
                "escalation": 0.02,
            }
        ]

        test_jsonl = tmp_path / "test.jsonl"
        test_jsonl.write_text('{"text": "hello", "label": "benign"}\n')

        with patch("firewall.classifier.evaluate.load_classifier", return_value=mock_clf):
            result = evaluate(
                model_path="dummy/path",
                test_path=test_jsonl,
                label_names=["benign", "injection", "jailbreak", "exfiltration", "escalation"],
            )

        assert "accuracy" in result
        assert "f1_macro" in result
        assert "confusion_matrix" in result
        assert "classification_report" in result

    def test_perfect_prediction_gives_accuracy_one(self, tmp_path) -> None:
        from unittest.mock import patch

        from firewall.classifier.evaluate import evaluate

        mock_clf = MagicMock()
        mock_clf.predict.return_value = [
            {
                "benign": 0.9,
                "injection": 0.05,
                "jailbreak": 0.02,
                "exfiltration": 0.02,
                "escalation": 0.01,
            }
        ]

        test_jsonl = tmp_path / "test.jsonl"
        test_jsonl.write_text('{"text": "hello world", "label": "benign"}\n')

        with patch("firewall.classifier.evaluate.load_classifier", return_value=mock_clf):
            result = evaluate(
                model_path="dummy/path",
                test_path=test_jsonl,
                label_names=["benign", "injection", "jailbreak", "exfiltration", "escalation"],
            )

        assert result["accuracy"] == 1.0


class TestEvaluateMain:
    def test_main_parses_config_and_prints_results(self, tmp_path) -> None:
        import yaml

        from firewall.classifier.evaluate import main

        config = {
            "output_dir": "dummy/model",
            "test_path": str(tmp_path / "test.jsonl"),
            "label_names": ["benign", "injection", "jailbreak", "exfiltration", "escalation"],
        }
        config_path = tmp_path / "eval_config.yaml"
        config_path.write_text(yaml.dump(config))

        mock_results = {
            "accuracy": 0.95,
            "f1_macro": 0.94,
            "f1_weighted": 0.95,
            "confusion_matrix": [[1]],
            "classification_report": "dummy report",
        }
        metrics_path = tmp_path / "metrics.json"

        with (
            patch("firewall.classifier.evaluate.evaluate", return_value=mock_results),
            patch(
                "sys.argv",
                ["evaluate", "--config", str(config_path), "--metrics-path", str(metrics_path)],
            ),
        ):
            main()

        import json

        written = json.loads(metrics_path.read_text())
        assert written["accuracy"] == 0.95
        assert written["confusion_matrix"] == [[1]]


class TestRobustnessMetrics:
    @staticmethod
    def _scores(**kw: float) -> dict[str, float]:
        base = {
            "benign": 0.0,
            "injection": 0.0,
            "jailbreak": 0.0,
            "exfiltration": 0.0,
            "escalation": 0.0,
        }
        base.update(kw)
        return base

    def test_threat_score_ignores_benign(self) -> None:
        from firewall.classifier.evaluate import threat_score

        scores = self._scores(benign=0.7, injection=0.2, jailbreak=0.1)
        assert threat_score(scores) == pytest.approx(0.2)

    def test_detection_rate_counts_non_clean(self) -> None:
        from firewall.classifier.evaluate import compute_robustness_metrics

        # One attack caught (threat_score 0.9 >= 0.3), one missed (0.05 < 0.3).
        preds = [
            self._scores(benign=0.1, injection=0.9),
            self._scores(benign=0.95, injection=0.05),
        ]
        result = compute_robustness_metrics(
            preds,
            ["injection", "injection"],
            ["base64", "roleplay"],
            clean_threshold=0.3,
            block_threshold=0.8,
        )
        assert result["n"] == 2
        assert result["detection_rate"] == pytest.approx(0.5)

    def test_block_rate_uses_block_threshold(self) -> None:
        from firewall.classifier.evaluate import compute_robustness_metrics

        # Both detected (>= 0.3) but only one hard-blocked (>= 0.8).
        preds = [self._scores(injection=0.9), self._scores(injection=0.5)]
        result = compute_robustness_metrics(
            preds,
            ["injection", "injection"],
            ["a", "b"],
            clean_threshold=0.3,
            block_threshold=0.8,
        )
        assert result["detection_rate"] == pytest.approx(1.0)
        assert result["block_rate"] == pytest.approx(0.5)

    def test_exact_class_accuracy(self) -> None:
        from firewall.classifier.evaluate import compute_robustness_metrics

        # First top-label matches true label, second does not.
        preds = [self._scores(injection=0.9), self._scores(jailbreak=0.9)]
        result = compute_robustness_metrics(
            preds,
            ["injection", "injection"],
            ["a", "b"],
            clean_threshold=0.3,
            block_threshold=0.8,
        )
        assert result["exact_class_accuracy"] == pytest.approx(0.5)

    def test_per_attack_type_breakdown(self) -> None:
        from firewall.classifier.evaluate import compute_robustness_metrics

        preds = [self._scores(injection=0.9), self._scores(benign=0.99, injection=0.01)]
        result = compute_robustness_metrics(
            preds,
            ["injection", "injection"],
            ["base64", "base64"],
            clean_threshold=0.3,
            block_threshold=0.8,
        )
        bucket = result["per_attack_type"]["base64"]
        assert bucket["n"] == 2
        assert bucket["detected"] == 1

    def test_empty_predictions_raises(self) -> None:
        from firewall.classifier.evaluate import compute_robustness_metrics

        with pytest.raises(ValueError, match="no predictions"):
            compute_robustness_metrics([], [], [], clean_threshold=0.3, block_threshold=0.8)


class TestEvaluateRobustness:
    def test_reads_set_and_returns_metrics(self, tmp_path: Any) -> None:
        from firewall.classifier.evaluate import evaluate_robustness

        mock_clf = MagicMock()
        mock_clf.predict.return_value = [
            {
                "benign": 0.1,
                "injection": 0.9,
                "jailbreak": 0.0,
                "exfiltration": 0.0,
                "escalation": 0.0,
            }
        ]
        data = tmp_path / "adv.jsonl"
        data.write_text('{"text": "x", "label": "injection", "attack_type": "base64"}\n')

        with patch("firewall.classifier.evaluate.load_classifier", return_value=mock_clf):
            result = evaluate_robustness(
                model_path="dummy",
                data_path=data,
                clean_threshold=0.3,
                block_threshold=0.8,
            )
        assert result["n"] == 1
        assert result["detection_rate"] == pytest.approx(1.0)

    def test_rejects_benign_label(self, tmp_path: Any) -> None:
        from firewall.classifier.evaluate import evaluate_robustness

        mock_clf = MagicMock()
        data = tmp_path / "adv.jsonl"
        data.write_text('{"text": "hi", "label": "benign", "attack_type": "none"}\n')

        with (
            patch("firewall.classifier.evaluate.load_classifier", return_value=mock_clf),
            pytest.raises(ValueError, match="only threats"),
        ):
            evaluate_robustness(
                model_path="dummy",
                data_path=data,
                clean_threshold=0.3,
                block_threshold=0.8,
            )


class TestRobustnessMain:
    def test_main_parses_args_and_logs_results(self, tmp_path: Any) -> None:
        from firewall.classifier.evaluate import robustness_main

        data_path = tmp_path / "adv.jsonl"
        data_path.write_text('{"text": "x", "label": "injection", "attack_type": "base64"}\n')

        mock_results = {
            "n": 1,
            "detection_rate": 1.0,
            "block_rate": 1.0,
            "exact_class_accuracy": 1.0,
            "per_attack_type": {"base64": {"n": 1, "detected": 1, "blocked": 1, "exact": 1}},
        }
        argv = [
            "robustness",
            "--model-path",
            "dummy/model",
            "--data-path",
            str(data_path),
        ]
        with (
            patch(
                "firewall.classifier.evaluate.evaluate_robustness", return_value=mock_results
            ) as mock_eval,
            patch("sys.argv", argv),
        ):
            robustness_main()

        mock_eval.assert_called_once()
        assert mock_eval.call_args.kwargs["clean_threshold"] == 0.3
        assert mock_eval.call_args.kwargs["block_threshold"] == 0.8


class TestSHAPExplainer:
    def test_explain_returns_list_with_expected_keys(self) -> None:
        from firewall.classifier.explain import SHAPExplainer

        mock_clf = MagicMock()

        def fake_predict(texts):
            return [
                {
                    "benign": 0.8,
                    "injection": 0.1,
                    "jailbreak": 0.05,
                    "exfiltration": 0.03,
                    "escalation": 0.02,
                }
                for _ in texts
            ]

        mock_clf.predict.side_effect = fake_predict

        explainer = SHAPExplainer(mock_clf, max_evals=10)
        result = explainer.explain(["ignore previous instructions"])

        assert isinstance(result, list)
        assert len(result) == 1
        assert "shap_values" in result[0]
        assert "tokens" in result[0]

    def test_explain_returns_one_entry_per_input(self) -> None:
        from firewall.classifier.explain import SHAPExplainer

        mock_clf = MagicMock()

        def fake_predict(texts):
            return [
                {
                    "benign": 0.8,
                    "injection": 0.1,
                    "jailbreak": 0.05,
                    "exfiltration": 0.03,
                    "escalation": 0.02,
                }
                for _ in texts
            ]

        mock_clf.predict.side_effect = fake_predict
        explainer = SHAPExplainer(mock_clf, max_evals=10)
        result = explainer.explain(["text one", "text two"])
        assert len(result) == 2
