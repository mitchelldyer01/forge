"""
Tests for forge.swarm.interaction — interaction selection for Round 2.

Mirrors: forge/swarm/interaction.py
"""

import json

import pytest

from forge.db.models import AgentPersona, SimulationTurn
from forge.swarm.interaction import compute_novelty_score, select_interactions


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
    reasoning: str = "test",
    key_concern: str = "",
) -> SimulationTurn:
    """Helper to create a SimulationTurn."""
    content = {"position": position, "reasoning": reasoning}
    if key_concern:
        content["key_concern"] = key_concern
    return SimulationTurn(
        id=st_id,
        simulation_id="s_test",
        round=round,
        agent_persona_id=agent_persona_id,
        turn_type="reaction",
        content=json.dumps(content),
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

    def test_select_picks_novel_argument(self):
        """Novel argument is selected over a non-novel high-contrarian turn."""
        agent = _make_persona("ap_me", "mainstream")
        my_turn = _make_turn("st_me", "ap_me", position="support")

        # Common reasoning shared by most agents
        common = "regulation will balance innovation with accountability"
        # Novel reasoning with unique vocabulary
        novel = "energy consumption of data centers will increase costs by 30%"

        turns = [
            my_turn,
            _make_turn("st_opp1", "ap_opp1", position="oppose",
                        confidence=70, reasoning=common),
            _make_turn("st_opp2", "ap_opp2", position="oppose",
                        confidence=60, reasoning=common),
            _make_turn("st_novel", "ap_novel", position="oppose",
                        confidence=50, reasoning=novel,
                        key_concern="energy infrastructure strain"),
        ]
        personas = {
            "ap_me": agent,
            "ap_opp1": _make_persona("ap_opp1", "conservative", contrarian=0.9),
            "ap_opp2": _make_persona("ap_opp2", "moderate", contrarian=0.5),
            "ap_novel": _make_persona("ap_novel", "environmentalist", contrarian=0.1),
        }

        selected = select_interactions(agent, my_turn, turns, personas, count=3)
        selected_ids = {t.id for t in selected}
        # The novel argument should be included despite low confidence/contrarian
        assert "st_novel" in selected_ids


@pytest.mark.unit
class TestMinorityDefense:
    def test_select_majority_agent_sees_least_common_position(self):
        """When majority agent has multiple opposing positions, novelty slot
        prefers the least-common position to prevent crowding it out.

        Setup: 8 oppose, 4 conditional, 2 support.
        For an oppose agent, opposing = 4 conditional + 2 support.
        The conditional agents have higher confidence and similar reasoning,
        so without minority defense, all 3 slots could go to conditional.
        The fix ensures the novelty slot picks from 'support' (least common).
        """
        agent = _make_persona("ap_me", "hawk")
        my_turn = _make_turn("st_me", "ap_me", position="oppose", confidence=70)

        turns = [my_turn]
        personas = {"ap_me": agent}
        # 7 more oppose agents
        for i in range(7):
            pid = f"ap_opp{i}"
            tid = f"st_opp{i}"
            turns.append(_make_turn(tid, pid, position="oppose", confidence=60 + i))
            personas[pid] = _make_persona(pid, f"hawk_{i}")
        # 4 conditional agents with high confidence
        common_reasoning = "phased implementation with regulatory sandbox"
        for i in range(4):
            pid = f"ap_cond{i}"
            tid = f"st_cond{i}"
            turns.append(_make_turn(
                tid, pid, position="conditional", confidence=80 + i,
                reasoning=common_reasoning,
            ))
            personas[pid] = _make_persona(pid, f"moderate_{i}")
        # 2 support agents with lower confidence and SAME reasoning
        # (so novelty won't differentiate them)
        for i in range(2):
            pid = f"ap_sup{i}"
            tid = f"st_sup{i}"
            turns.append(_make_turn(
                tid, pid, position="support", confidence=55 + i,
                reasoning=common_reasoning,
            ))
            personas[pid] = _make_persona(pid, f"dove_{i}")

        selected = select_interactions(agent, my_turn, turns, personas, count=3)
        selected_positions = [t.position for t in selected]
        # Must include at least one support (the least-common opposing position)
        assert "support" in selected_positions

    def test_minority_agent_selection_unchanged(self):
        """Minority agents still get normal selection (opposing = majority views)."""
        agent = _make_persona("ap_me", "dove")
        my_turn = _make_turn("st_me", "ap_me", position="support", confidence=60)

        # 8 oppose, 2 support (including me)
        turns = [my_turn]
        personas = {"ap_me": agent}
        for i in range(8):
            pid = f"ap_opp{i}"
            tid = f"st_opp{i}"
            turns.append(_make_turn(tid, pid, position="oppose", confidence=60 + i))
            personas[pid] = _make_persona(pid, f"hawk_{i}")
        pid = "ap_sup1"
        tid = "st_sup1"
        turns.append(_make_turn(tid, pid, position="support", confidence=55))
        personas[pid] = _make_persona(pid, "dove_1")

        selected = select_interactions(agent, my_turn, turns, personas, count=3)
        # Minority agent should see opposing (majority) views — all should be oppose
        assert all(t.position == "oppose" for t in selected)


@pytest.mark.unit
class TestComputeNoveltyScore:
    def test_unique_words_score_higher(self):
        """A turn with unusual vocabulary scores higher than common vocabulary."""
        common = "regulation will balance innovation with accountability"
        novel = "energy consumption of data centers will increase electricity costs"

        common_turns = [
            _make_turn(f"st_{i}", f"ap_{i}", reasoning=common)
            for i in range(5)
        ]
        novel_turn = _make_turn("st_novel", "ap_novel", reasoning=novel)
        all_turns = common_turns + [novel_turn]

        novel_score = compute_novelty_score(novel_turn, all_turns)
        common_score = compute_novelty_score(common_turns[0], all_turns)
        assert novel_score > common_score

    def test_empty_reasoning_returns_zero(self):
        """Turn with empty reasoning gets a novelty score of 0.0."""
        turn = _make_turn("st_empty", "ap_empty", reasoning="")
        all_turns = [
            turn,
            _make_turn("st_other", "ap_other", reasoning="some content here"),
        ]
        assert compute_novelty_score(turn, all_turns) == 0.0

    def test_stop_words_ignored(self):
        """Common stop words don't inflate novelty scores."""
        # Turn with only stop words
        stop_turn = _make_turn("st_stop", "ap_stop",
                               reasoning="the and is of to in for")
        other = _make_turn("st_other", "ap_other",
                           reasoning="quantum computing revolutionizes cryptography")
        all_turns = [stop_turn, other]
        assert compute_novelty_score(stop_turn, all_turns) == 0.0
