"""Unit tests for the LangGraph orchestration graph.

Covers:
- CLEAN zone: top_score < clean_threshold → PASS without invoking judge
- GRAY  zone: score in the ambiguous band → judge node called
- BLOCK zone: top_score ≥ block_threshold → BLOCK without invoking judge
- Correct state mutations across node transitions
"""
from __future__ import annotations

import pytest


class TestZoneRouting:
    """Tests for zone-based routing in :mod:`firewall.orchestrator.graph`."""

    def test_clean_zone_routes_to_execute_node(self) -> None:
        raise NotImplementedError

    def test_block_zone_routes_to_log_node(self) -> None:
        raise NotImplementedError

    def test_gray_zone_routes_to_judge_node(self) -> None:
        raise NotImplementedError

    def test_clean_zone_does_not_invoke_judge(self) -> None:
        raise NotImplementedError


class TestNodes:
    """Tests for individual node functions in :mod:`firewall.orchestrator.nodes`."""

    def test_classify_node_sets_zone_field(self) -> None:
        raise NotImplementedError

    def test_log_node_appends_entry_to_logs(self) -> None:
        raise NotImplementedError

    def test_execute_node_sets_pass_decision(self) -> None:
        raise NotImplementedError
