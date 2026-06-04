"""Unit tests for the training entrypoint and attention-heatmap plotting.

Both are exercised with mocked HuggingFace internals so no model is downloaded
or trained — the point is to verify wiring (config -> TrainingArguments) and that
the plotting helper produces a figure.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
import yaml
from firewall.classifier.explain import plot_attention_heatmap
from firewall.classifier.train import train


def test_train_wires_config_into_training_arguments(tmp_path) -> None:
    config = {
        "label_names": ["benign", "injection"],
        "num_labels": 2,
        "model_name": "dummy-model",
        "train_path": "train.jsonl",
        "val_path": "val.jsonl",
        "max_length": 128,
        "output_dir": str(tmp_path / "out"),
        "num_epochs": 3,
        "batch_size": 8,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "fp16": False,
        "seed": 42,
        "early_stopping_patience": 2,
    }
    config_path = tmp_path / "training.yaml"
    config_path.write_text(yaml.safe_dump(config))

    with (
        patch("firewall.classifier.train.AutoModelForSequenceClassification"),
        patch("firewall.classifier.train.AutoTokenizer"),
        patch("firewall.classifier.train.FirewallDataset"),
        patch(
            "firewall.classifier.train._load_jsonl",
            return_value=(["clean text", "ignore previous"], [0, 1]),
        ),
        patch("firewall.classifier.train.WeightedTrainer") as mock_trainer,
    ):
        train(config_path)

    # TrainingArguments built from the config and handed to the trainer.
    args = mock_trainer.call_args.kwargs["args"]
    assert args.num_train_epochs == 3
    assert args.per_device_train_batch_size == 8
    assert args.learning_rate == 2e-5
    assert args.seed == 42

    # Training actually ran and the model was persisted.
    mock_trainer.return_value.train.assert_called_once()
    mock_trainer.return_value.save_model.assert_called_once()


def test_plot_attention_heatmap_saves_figure(tmp_path) -> None:
    model = MagicMock()
    model.tokenizer.tokenize.return_value = ["alpha", "beta", "gamma"]
    encoding = {"input_ids": torch.zeros(1, 3, dtype=torch.long)}
    model.tokenizer.return_value.to.return_value = encoding

    outputs = MagicMock()
    outputs.attentions = [torch.rand(1, 4, 5, 5)]  # one layer, 4 heads, 5x5 attention
    model.model.return_value = outputs

    out_path = tmp_path / "attention.png"
    result = plot_attention_heatmap("alpha beta gamma", model, output_path=out_path)

    assert result == out_path
    assert out_path.exists()
