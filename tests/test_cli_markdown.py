"""
Tests for `forge/cli_markdown.py` — markdown rendering for simulation turns.

Mirrors: forge/cli_markdown.py
"""

import json
import re

import pytest

from forge.db.models import Simulation


def _make_sim(**overrides) -> Simulation:
    """Create a minimal Simulation for testing."""
    defaults = {
        "id": "s_01TESTID1234567890",
        "mode": "scenario",
        "seed_text": "What if the EU regulates AI agents?",
        "agent_count": 3,
        "rounds": 3,
        "status": "complete",
        "started_at": "2026-03-18T10:00:00Z",
    }
    defaults.update(overrides)
    return Simulation(**defaults)


def _make_row(
    round_num: int,
    archetype: str,
    position: str,
    confidence: int,
    turn_type: str = "reaction",
    content: dict | None = None,
) -> dict:
    """Create a turn row dict matching store.list_turns_with_agent() output."""
    if content is None:
        content = {
            "position": position,
            "confidence": confidence,
            "reasoning": f"Reasoning from {archetype} in round {round_num}",
        }
        if round_num == 1:
            content["key_concern"] = f"Concern from {archetype}"
            content["key_insight"] = f"Insight from {archetype}"
        elif round_num == 2:
            content["key_point"] = f"Key point from {archetype}"
        elif round_num == 3:
            content["final_position"] = position
            content["conviction_delta"] = 5
            content["changed_mind"] = False
            content["key_insight"] = f"Final insight from {archetype}"
    return {
        "round": round_num,
        "archetype": archetype,
        "position": position,
        "confidence": confidence,
        "turn_type": turn_type,
        "content": json.dumps(content),
        "agent_persona_id": f"ap_{archetype}",
        "persona_json": json.dumps({"name": archetype.replace("_", " ").title()}),
    }


def _sample_rows() -> list[dict]:
    """Build a 3-round, 2-agent sample dataset."""
    return [
        _make_row(1, "tech_optimist", "support", 80),
        _make_row(1, "policy_analyst", "oppose", 65),
        _make_row(2, "tech_optimist", "support", 75, turn_type="challenge"),
        _make_row(2, "policy_analyst", "conditional", 70, turn_type="concession"),
        _make_row(3, "tech_optimist", "support", 85, turn_type="convergence"),
        _make_row(3, "policy_analyst", "conditional", 75, turn_type="convergence"),
    ]


@pytest.mark.unit
class TestRenderTurnsMarkdown:
    def test_render_turns_markdown_returns_string(self):
        from forge.cli_markdown import render_turns_markdown

        result = render_turns_markdown(_sample_rows(), _make_sim())
        assert isinstance(result, str)
        assert len(result) > 0

    def test_render_turns_markdown_includes_simulation_metadata(self):
        from forge.cli_markdown import render_turns_markdown

        sim = _make_sim()
        result = render_turns_markdown(_sample_rows(), sim)
        assert sim.id in result
        assert sim.seed_text in result

    def test_render_turns_markdown_includes_round_headers(self):
        from forge.cli_markdown import render_turns_markdown

        result = render_turns_markdown(_sample_rows(), _make_sim())
        assert "## Round 1" in result
        assert "## Round 2" in result
        assert "## Round 3" in result

    def test_render_turns_markdown_includes_agent_positions(self):
        from forge.cli_markdown import render_turns_markdown

        result = render_turns_markdown(_sample_rows(), _make_sim())
        assert "tech_optimist" in result
        assert "support" in result
        assert "policy_analyst" in result

    def test_render_turns_markdown_includes_confidence(self):
        from forge.cli_markdown import render_turns_markdown

        result = render_turns_markdown(_sample_rows(), _make_sim())
        assert "80%" in result
        assert "65%" in result

    def test_render_turns_markdown_includes_reasoning(self):
        from forge.cli_markdown import render_turns_markdown

        result = render_turns_markdown(_sample_rows(), _make_sim())
        assert "Reasoning from tech_optimist in round 1" in result
        assert "Reasoning from policy_analyst in round 2" in result

    def test_render_turns_markdown_round3_conviction_delta(self):
        from forge.cli_markdown import render_turns_markdown

        result = render_turns_markdown(_sample_rows(), _make_sim())
        assert "conviction_delta" in result.lower() or "Conviction delta" in result

    def test_render_turns_markdown_includes_consensus(self):
        from forge.cli_markdown import render_turns_markdown

        result = render_turns_markdown(_sample_rows(), _make_sim())
        assert "## Consensus" in result

    def test_render_turns_markdown_empty_rows(self):
        from forge.cli_markdown import render_turns_markdown

        result = render_turns_markdown([], _make_sim())
        assert isinstance(result, str)
        assert _make_sim().seed_text in result
        assert "## Round" not in result

    def test_render_turns_markdown_malformed_json(self):
        from forge.cli_markdown import render_turns_markdown

        rows = [
            {
                "round": 1,
                "archetype": "analyst",
                "position": "support",
                "confidence": 70,
                "turn_type": "reaction",
                "content": "NOT VALID JSON {{{",
                "agent_persona_id": "ap_analyst",
                "persona_json": "{}",
            }
        ]
        result = render_turns_markdown(rows, _make_sim())
        assert isinstance(result, str)
        assert "analyst" in result

    def test_render_turns_markdown_no_box_drawing(self):
        from forge.cli_markdown import render_turns_markdown

        result = render_turns_markdown(_sample_rows(), _make_sim())
        box_drawing = re.compile(r"[\u2500-\u257F]")
        assert not box_drawing.search(result), "Output contains box-drawing characters"

    def test_render_turn_r2_shows_steel_man(self):
        from forge.cli_markdown import render_turns_markdown

        content = {
            "turn_type": "challenge",
            "position": "oppose",
            "confidence": 70,
            "reasoning": "Test reasoning",
            "steel_man": "The strongest argument for the other side is X",
            "key_point": "My main point",
        }
        rows = [_make_row(2, "analyst", "oppose", 70, turn_type="challenge", content=content)]
        result = render_turns_markdown(rows, _make_sim())
        assert "**Steel-man:**" in result
        assert "strongest argument for the other side is X" in result

    def test_render_turn_r2_shows_concrete_mechanism(self):
        from forge.cli_markdown import render_turns_markdown

        content = {
            "turn_type": "refine",
            "position": "conditional",
            "confidence": 65,
            "reasoning": "Test reasoning",
            "concrete_mechanism": "Require annual audits with €500K penalty threshold",
            "key_point": "My point",
        }
        rows = [_make_row(
            2, "policy_analyst", "conditional", 65,
            turn_type="refine", content=content,
        )]
        result = render_turns_markdown(rows, _make_sim())
        assert "**Mechanism:**" in result
        assert "annual audits" in result

    def test_render_turn_r2_skips_none_mechanism(self):
        from forge.cli_markdown import render_turns_markdown

        content = {
            "turn_type": "challenge",
            "position": "oppose",
            "confidence": 70,
            "reasoning": "Test reasoning",
            "concrete_mechanism": "none",
            "key_point": "My point",
        }
        rows = [_make_row(2, "critic", "oppose", 70, turn_type="challenge", content=content)]
        result = render_turns_markdown(rows, _make_sim())
        assert "**Mechanism:**" not in result

    def test_render_turn_r3_shows_confidence_justification(self):
        from forge.cli_markdown import render_turns_markdown

        content = {
            "final_position": "oppose",
            "confidence": 85,
            "conviction_delta": 10,
            "changed_mind": False,
            "reasoning": "Final synthesis",
            "confidence_justification": "Legal precedent from MiFID II compliance failures",
            "key_insight": "Test insight",
        }
        rows = [_make_row(
            3, "legal_expert", "oppose", 85,
            turn_type="convergence", content=content,
        )]
        result = render_turns_markdown(rows, _make_sim())
        assert "**Confidence basis:**" in result
        assert "MiFID II" in result
