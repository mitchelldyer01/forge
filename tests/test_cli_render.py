"""Tests for forge/cli_render.py — simulation rendering helpers."""

from __future__ import annotations

import pytest

from forge.cli_render import PositionStats, RoundSummary, summarize_round


def _make_turn(position: str, confidence: int, agent_id: str = "a1"):
    """Create a minimal SimulationTurn-like object for testing."""
    from types import SimpleNamespace

    return SimpleNamespace(
        position=position,
        confidence=confidence,
        agent_persona_id=agent_id,
    )


@pytest.mark.unit
class TestSummarizeRound:
    def test_summarize_round_counts_positions(self) -> None:
        """Position counts are correct for a mix of positions."""
        turns = [
            _make_turn("support", 80, "a1"),
            _make_turn("support", 90, "a2"),
            _make_turn("oppose", 60, "a3"),
            _make_turn("conditional", 70, "a4"),
            _make_turn("conditional", 80, "a5"),
        ]
        summary = summarize_round(turns, round_num=1, expected_agents=5)
        assert summary.round_num == 1
        assert summary.responded == 5
        assert summary.failed == 0
        assert summary.positions["support"].count == 2
        assert summary.positions["support"].avg_confidence == 85.0
        assert summary.positions["oppose"].count == 1
        assert summary.positions["oppose"].avg_confidence == 60.0
        assert summary.positions["conditional"].count == 2

    def test_summarize_round_tracks_failures(self) -> None:
        """When fewer turns than expected agents, reports failures."""
        turns = [
            _make_turn("support", 80, "a1"),
            _make_turn("support", 90, "a2"),
        ]
        summary = summarize_round(turns, round_num=3, expected_agents=5)
        assert summary.responded == 2
        assert summary.failed == 3

    def test_summarize_round_detects_changed_positions(self) -> None:
        """Agents whose position differs from prev_turns are counted."""
        prev = [
            _make_turn("support", 80, "a1"),
            _make_turn("oppose", 60, "a2"),
            _make_turn("conditional", 70, "a3"),
        ]
        current = [
            _make_turn("support", 85, "a1"),   # same
            _make_turn("support", 75, "a2"),   # changed
            _make_turn("oppose", 65, "a3"),    # changed
        ]
        summary = summarize_round(
            current, round_num=2, expected_agents=3, prev_turns=prev,
        )
        assert summary.changed == 2

    def test_summarize_round_empty_turns(self) -> None:
        """All agents failed — responded=0, failed=expected."""
        summary = summarize_round([], round_num=1, expected_agents=3)
        assert summary.responded == 0
        assert summary.failed == 3
        assert summary.positions == {}

    def test_summarize_round_no_prev_turns_zero_changed(self) -> None:
        """Without prev_turns, changed is always 0."""
        turns = [_make_turn("support", 80, "a1")]
        summary = summarize_round(turns, round_num=1, expected_agents=1)
        assert summary.changed == 0


@pytest.mark.unit
class TestRoundSummaryFormat:
    def test_format_line_basic(self) -> None:
        """format_line produces expected structure."""
        summary = RoundSummary(
            round_num=1,
            responded=21,
            expected_agents=21,
            failed=0,
            changed=0,
            positions={
                "conditional": PositionStats(count=17, avg_confidence=79.0),
                "support": PositionStats(count=3, avg_confidence=88.0),
                "oppose": PositionStats(count=1, avg_confidence=70.0),
            },
        )
        line = summary.format_line()
        assert "Round 1" in line
        assert "21/21" in line
        assert "conditional" in line

    def test_format_line_with_failures(self) -> None:
        """format_line includes failure count."""
        summary = RoundSummary(
            round_num=3,
            responded=19,
            expected_agents=21,
            failed=2,
            changed=1,
            positions={
                "support": PositionStats(count=19, avg_confidence=80.0),
            },
        )
        line = summary.format_line()
        assert "2 failed" in line
        assert "1 changed" in line
