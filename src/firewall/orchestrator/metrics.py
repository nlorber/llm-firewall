"""Prometheus metrics for the firewall orchestrator pipeline."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram

REGISTRY = CollectorRegistry()

classify_duration = Histogram(
    "firewall_classify_duration_seconds",
    "Classifier inference latency in seconds",
    labelnames=["zone"],
    registry=REGISTRY,
)

judge_duration = Histogram(
    "firewall_judge_duration_seconds",
    "LLM judge call latency in seconds",
    labelnames=["decision"],
    registry=REGISTRY,
)

requests_total = Counter(
    "firewall_requests_total",
    "Total analyzed requests",
    labelnames=["zone", "final_decision"],
    registry=REGISTRY,
)

classification_label_total = Counter(
    "firewall_classification_label_total",
    "Classification counts by label",
    labelnames=["label"],
    registry=REGISTRY,
)

judge_tier_total = Counter(
    "firewall_judge_tier_total",
    "Tiered-judge verdicts by tier that answered (local kept vs escalated to claude)",
    labelnames=["tier"],
    registry=REGISTRY,
)
