"""
Tests for forge.swarm.debate_digest — debate digest for convergence context.

Mirrors: forge/swarm/debate_digest.py
"""

import json

import pytest

from forge.db.models import AgentPersona, SimulationTurn
from forge.swarm.debate_digest import build_debate_digest


def _make_persona(ap_id: str, archetype: str) -> AgentPersona:
    """Helper to create an AgentPersona."""
    persona = {"archetype": archetype, "name": f"Agent {ap_id}"}
    return AgentPersona(
        id=ap_id,
        archetype=archetype,
        persona_json=json.dumps(persona),
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )


def _make_turn(
    st_id: str,
    agent_persona_id: str,
    *,
    position: str = "support",
    key_point: str = "test point",
    reasoning: str = "test reasoning",
) -> SimulationTurn:
    """Helper to create a Round 2 SimulationTurn."""
    content = {
        "position": position,
        "reasoning": reasoning,
        "key_point": key_point,
    }
    return SimulationTurn(
        id=st_id,
        simulation_id="s_test",
        round=2,
        agent_persona_id=agent_persona_id,
        turn_type="challenge",
        content=json.dumps(content),
        position=position,
        confidence=70,
        created_at="2026-01-01T00:00:00+00:00",
    )


@pytest.mark.unit
class TestBuildDebateDigest:
    def test_excludes_own_turn(self):
        """The agent's own R2 turn is not in the digest."""
        personas = {
            "ap_me": _make_persona("ap_me", "analyst"),
            "ap_other": _make_persona("ap_other", "critic"),
        }
        turns = [
            _make_turn("st_me", "ap_me", key_point="my point"),
            _make_turn("st_other", "ap_other", key_point="other point"),
        ]
        digest = build_debate_digest(turns, "ap_me", personas)
        assert "my point" not in digest
        assert "other point" in digest

    def test_respects_max_items(self):
        """Returns at most max_items entries."""
        personas = {}
        turns = []
        for i in range(10):
            ap_id = f"ap_{i}"
            personas[ap_id] = _make_persona(ap_id, f"type_{i}")
            turns.append(_make_turn(
                f"st_{i}", ap_id,
                key_point=f"point {i}",
                reasoning=f"unique argument number {i} about topic {i}",
            ))

        digest = build_debate_digest(turns, "ap_exclude", personas, max_items=3)
        # Count lines starting with "- " (each digest entry)
        entries = [line for line in digest.strip().split("\n") if line.startswith("- ")]
        assert len(entries) <= 3

    def test_empty_turns(self):
        """Empty turns list returns empty string."""
        digest = build_debate_digest([], "ap_me", {})
        assert digest == ""

    def test_digest_includes_all_represented_positions(self):
        """Digest includes at least one entry from each position."""
        personas = {}
        turns = []
        # 7 support turns with high novelty vocabulary
        for i in range(7):
            ap_id = f"ap_sup{i}"
            personas[ap_id] = _make_persona(ap_id, f"supporter_{i}")
            turns.append(_make_turn(
                f"st_sup{i}", ap_id,
                position="support",
                key_point=f"unique support point {i}",
                reasoning=f"novel argument about topic {i} regarding aspect {i}",
            ))
        # 3 oppose turns with less novel vocabulary
        for i in range(3):
            ap_id = f"ap_opp{i}"
            personas[ap_id] = _make_persona(ap_id, f"opposer_{i}")
            turns.append(_make_turn(
                f"st_opp{i}", ap_id,
                position="oppose",
                key_point=f"opposition point {i}",
                reasoning="common regulation argument about oversight",
            ))

        digest = build_debate_digest(turns, "ap_exclude", personas, max_items=5)
        assert "oppose" in digest, "Digest missing oppose position"

    def test_digest_philosopher_capture_prevented(self):
        """One articulate agent doesn't dominate the entire digest."""
        personas = {}
        turns = []
        # 5 conditional turns with similar reasoning
        for i in range(5):
            ap_id = f"ap_cond{i}"
            personas[ap_id] = _make_persona(ap_id, f"moderate_{i}")
            turns.append(_make_turn(
                f"st_cond{i}", ap_id,
                position="conditional",
                key_point=f"conditional point {i}",
                reasoning="phased implementation with regulatory sandbox",
            ))
        # 1 oppose turn with very unique vocabulary
        ap_id = "ap_philosopher"
        personas[ap_id] = _make_persona(ap_id, "philosopher")
        turns.append(_make_turn(
            "st_phil", ap_id,
            position="oppose",
            key_point="epistemic humility demands restraint",
            reasoning="deontological framework compels precautionary abstention",
        ))

        digest = build_debate_digest(turns, "ap_exclude", personas, max_items=5)
        # Must include at least one conditional entry, not just the philosopher
        assert "conditional" in digest

    def test_digest_single_position_still_works(self):
        """Digest works when all turns share the same position."""
        personas = {}
        turns = []
        for i in range(5):
            ap_id = f"ap_{i}"
            personas[ap_id] = _make_persona(ap_id, f"type_{i}")
            turns.append(_make_turn(
                f"st_{i}", ap_id,
                position="conditional",
                key_point=f"point {i}",
                reasoning=f"unique argument number {i} about aspect {i}",
            ))

        digest = build_debate_digest(turns, "ap_exclude", personas, max_items=5)
        entries = [line for line in digest.strip().split("\n") if line.startswith("- ")]
        assert len(entries) >= 1

    def test_formats_archetype_and_key_point(self):
        """Output contains archetype names and key points."""
        personas = {
            "ap_env": _make_persona("ap_env", "environmental_economist"),
        }
        turns = [
            _make_turn("st_env", "ap_env",
                        position="conditional",
                        key_point="energy costs will increase by 30%"),
        ]
        digest = build_debate_digest(turns, "ap_me", personas)
        assert "environmental_economist" in digest
        assert "energy costs will increase by 30%" in digest
        assert "conditional" in digest
