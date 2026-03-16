"""
Tests for forge.swarm.interaction — interaction selection for Round 2.

Mirrors: forge/swarm/interaction.py
"""

import json

import pytest

from forge.db.models import AgentPersona, SimulationTurn
from forge.swarm.interaction import select_interactions


def _make_persona(
    ap_id: str,
    archetype: str,
    *,
    contrarian: float = 0.3,
    expertise: list[str] | None = None,
) -> AgentPersona:
    """Helper to create an AgentPersona with personality traits."""
    persona = {
        "archetype": archetype,
        "name": f"Agent {ap_id}",
        "expertise": expertise or [],
        "personality": {
            "risk_appetite": "medium",
            "optimism_bias": "realist",
            "contrarian_tendency": contrarian,
            "analytical_depth": "deep",
        },
    }
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
    confidence: int = 70,
    round: int = 1,
) -> SimulationTurn:
    """Helper to create a SimulationTurn."""
    return SimulationTurn(
        id=st_id,
        simulation_id="s_test",
        round=round,
        agent_persona_id=agent_persona_id,
        turn_type="reaction",
        content=json.dumps({"position": position, "reasoning": "test"}),
        position=position,
        confidence=confidence,
        created_at="2026-01-01T00:00:00+00:00",
    )


@pytest.mark.unit
class TestSelectInteractions:
    def test_select_strongest_opposing(self):
        """Selects the highest-confidence opposing turn first."""
        agent = _make_persona("ap_me", "optimist")
        my_turn = _make_turn("st_me", "ap_me", position="support", confidence=60)

        turns = [
            my_turn,
            _make_turn("st_opp1", "ap_opp1", position="oppose", confidence=50),
            _make_turn("st_opp2", "ap_opp2", position="oppose", confidence=90),
            _make_turn("st_ally", "ap_ally", position="support", confidence=80),
        ]
        personas = {
            "ap_me": agent,
            "ap_opp1": _make_persona("ap_opp1", "pessimist"),
            "ap_opp2": _make_persona("ap_opp2", "hawk"),
            "ap_ally": _make_persona("ap_ally", "bull"),
        }

        selected = select_interactions(agent, my_turn, turns, personas, count=1)
        assert len(selected) == 1
        # Should pick the highest-confidence opposing turn
        assert selected[0].id == "st_opp2"

    def test_select_contrarian(self):
        """Includes the most contrarian opposing agent."""
        agent = _make_persona("ap_me", "mainstream")
        my_turn = _make_turn("st_me", "ap_me", position="support")

        turns = [
            my_turn,
            _make_turn("st_opp1", "ap_opp1", position="oppose", confidence=70),
            _make_turn("st_opp2", "ap_opp2", position="oppose", confidence=70),
        ]
        personas = {
            "ap_me": agent,
            "ap_opp1": _make_persona("ap_opp1", "mild", contrarian=0.2),
            "ap_opp2": _make_persona("ap_opp2", "rebel", contrarian=0.9),
        }

        selected = select_interactions(agent, my_turn, turns, personas, count=2)
        assert len(selected) == 2
        # Both opposing are selected; the contrarian should be among them
        selected_ids = {t.id for t in selected}
        assert "st_opp2" in selected_ids

    def test_select_domain_expert_disagreement(self):
        """Includes an opposing domain expert when available."""
        agent = _make_persona("ap_me", "generalist")
        my_turn = _make_turn("st_me", "ap_me", position="support")

        turns = [
            my_turn,
            _make_turn("st_opp1", "ap_opp1", position="oppose", confidence=60),
            _make_turn("st_opp2", "ap_opp2", position="oppose", confidence=60),
            _make_turn("st_opp3", "ap_opp3", position="oppose", confidence=60),
        ]
        personas = {
            "ap_me": agent,
            "ap_opp1": _make_persona("ap_opp1", "layperson"),
            "ap_opp2": _make_persona("ap_opp2", "expert", expertise=["ai", "policy"]),
            "ap_opp3": _make_persona("ap_opp3", "bystander"),
        }

        selected = select_interactions(agent, my_turn, turns, personas, count=3)
        selected_ids = {t.id for t in selected}
        # The domain expert should be included
        assert "st_opp2" in selected_ids

    def test_select_excludes_self(self):
        """The agent's own turn is never in the selection."""
        agent = _make_persona("ap_me", "analyst")
        my_turn = _make_turn("st_me", "ap_me", position="support", confidence=99)

        turns = [
            my_turn,
            _make_turn("st_other", "ap_other", position="oppose", confidence=50),
        ]
        personas = {
            "ap_me": agent,
            "ap_other": _make_persona("ap_other", "critic"),
        }

        selected = select_interactions(agent, my_turn, turns, personas, count=3)
        assert all(t.id != "st_me" for t in selected)

    def test_select_all_agree_fallback(self):
        """When all agents agree, selects those with highest contrarian tendency."""
        agent = _make_persona("ap_me", "consensus")
        my_turn = _make_turn("st_me", "ap_me", position="support")

        turns = [
            my_turn,
            _make_turn("st_a", "ap_a", position="support", confidence=80),
            _make_turn("st_b", "ap_b", position="support", confidence=60),
        ]
        personas = {
            "ap_me": agent,
            "ap_a": _make_persona("ap_a", "follower", contrarian=0.1),
            "ap_b": _make_persona("ap_b", "rebel", contrarian=0.9),
        }

        selected = select_interactions(agent, my_turn, turns, personas, count=2)
        # Should still return something (fallback to most contrarian regardless)
        assert len(selected) >= 1
        assert selected[0].id == "st_b"  # highest contrarian tendency

    def test_select_fewer_than_count(self):
        """Returns fewer than count when not enough opposing views exist."""
        agent = _make_persona("ap_me", "analyst")
        my_turn = _make_turn("st_me", "ap_me", position="support")

        turns = [
            my_turn,
            _make_turn("st_opp", "ap_opp", position="oppose", confidence=60),
        ]
        personas = {
            "ap_me": agent,
            "ap_opp": _make_persona("ap_opp", "critic"),
        }

        selected = select_interactions(agent, my_turn, turns, personas, count=3)
        assert len(selected) == 1

    def test_select_empty_turns(self):
        """Empty turns list returns empty selection."""
        agent = _make_persona("ap_me", "analyst")
        my_turn = _make_turn("st_me", "ap_me", position="support")

        selected = select_interactions(agent, my_turn, [], {}, count=3)
        assert selected == []
