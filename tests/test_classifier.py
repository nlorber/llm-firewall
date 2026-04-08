"""Unit tests for the firewall classifier.

Covers:
- FirewallClassifier output format and probability normalisation
- Batch inference correctness
- Edge cases: empty-ish string, very long prompt, Unicode
- FirewallDataset length and item structure
"""
from __future__ import annotations

import pytest


class TestFirewallClassifier:
    """Tests for :class:`firewall.classifier.model.FirewallClassifier`."""

    def test_predict_returns_list_of_dicts(self) -> None:
        raise NotImplementedError

    def test_probabilities_sum_to_one(self) -> None:
        raise NotImplementedError

    def test_all_label_names_present_in_output(self) -> None:
        raise NotImplementedError

    def test_batch_inference_returns_one_entry_per_input(self) -> None:
        raise NotImplementedError


class TestFirewallDataset:
    """Tests for :class:`firewall.classifier.dataset.FirewallDataset`."""

    def test_dataset_length_matches_input(self) -> None:
        raise NotImplementedError

    def test_item_contains_required_tensor_keys(self) -> None:
        raise NotImplementedError
