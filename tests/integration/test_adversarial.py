"""Adversarial robustness evaluation: obfuscated and evasive attack prompts.

Skipped in CI (model checkpoint is gitignored). Run locally after `make train`.
Tests are grouped by attack type. The encoding/language-barrier categories (base64,
unicode, multilingual) were historically the hard cases; the current classifier catches
them, and the LLM judge remains the safety net if a retrain regresses on them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

MODEL_PATH = Path("models/classifier")
ADVERSARIAL_PATH = Path("data/adversarial/adversarial_prompts.jsonl")
SKIP_REASON = "model checkpoint not found (run `make train` first)"

# Encoding/language-barrier attack types — historically the hardest for a token-level
# classifier. The current model detects them (see test_encoded_attacks_detected); the LLM
# judge is the fallback if a retrain regresses on them.
EXPECTED_HARD_TYPES = {"base64_encoding", "multilingual", "unicode_homoglyph"}


def _load_adversarial() -> list[dict[str, str]]:
    entries = []
    with ADVERSARIAL_PATH.open() as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def _classify(classifier: object, text: str) -> tuple[str, float]:
    """Return (top_label, threat_score) for a single prompt."""
    results = classifier.predict([text])  # type: ignore[union-attr]
    scores = results[0]
    top_label = max(scores, key=scores.__getitem__)
    threat_score = max((v for k, v in scores.items() if k != "benign"), default=0.0)
    return top_label, threat_score


@pytest.mark.skipif(not MODEL_PATH.exists(), reason=SKIP_REASON)
class TestAdversarialRobustness:
    """Run adversarial prompts through the classifier and verify detection."""

    @pytest.fixture(scope="class")
    def classifier(self):
        from firewall.classifier.model import load_classifier

        return load_classifier(str(MODEL_PATH))

    @pytest.fixture(scope="class")
    def adversarial_prompts(self) -> list[dict[str, str]]:
        return _load_adversarial()

    # --- Standard obfuscation (classifier should catch these) ---

    @pytest.mark.parametrize(
        "attack_type",
        [
            "payload_splitting",
            "persona_roleplay",
            "instruction_nesting",
            "code_injection",
            "case_manipulation",
            "semantic_obfuscation",
        ],
    )
    def test_standard_obfuscation_detected(
        self,
        classifier,
        adversarial_prompts: list[dict[str, str]],
        attack_type: str,
    ) -> None:
        """Prompts with text-level obfuscation should not be classified as benign."""
        prompts = [p for p in adversarial_prompts if p["attack_type"] == attack_type]
        assert prompts, f"no prompts found for attack_type={attack_type}"
        for prompt in prompts:
            top_label, threat_score = _classify(classifier, prompt["text"])
            assert (
                top_label != "benign"
            ), f"[{attack_type}] misclassified as benign: {prompt['text'][:80]}..."

    # --- Historically hard categories (encoding/language barriers) ---

    @pytest.mark.parametrize("attack_type", sorted(EXPECTED_HARD_TYPES))
    def test_encoded_attacks_detected(
        self,
        classifier,
        adversarial_prompts: list[dict[str, str]],
        attack_type: str,
    ) -> None:
        """Base64, unicode, and multilingual attacks — historically evasive, now caught by the
        classifier (the LLM judge is the fallback if a retrain regresses)."""
        prompts = [p for p in adversarial_prompts if p["attack_type"] == attack_type]
        assert prompts, f"no prompts found for attack_type={attack_type}"
        for prompt in prompts:
            top_label, threat_score = _classify(classifier, prompt["text"])
            assert (
                top_label != "benign"
            ), f"[{attack_type}] misclassified as benign: {prompt['text'][:80]}..."

    # --- Aggregate detection rate ---

    def test_overall_detection_rate_above_50_percent(
        self,
        classifier,
        adversarial_prompts: list[dict[str, str]],
    ) -> None:
        """At least half of all adversarial prompts should be flagged as non-benign."""
        detected = 0
        for prompt in adversarial_prompts:
            top_label, _ = _classify(classifier, prompt["text"])
            if top_label != "benign":
                detected += 1
        rate = detected / len(adversarial_prompts)
        assert (
            rate >= 0.5
        ), f"detection rate {rate:.0%} ({detected}/{len(adversarial_prompts)}) is below 50%"
