"""
Tests for forge.swarm.consensus — computational consensus extraction.

Mirrors: forge/swarm/consensus.py
"""

import json

import pytest

from forge.db.models import AgentPersona, SimulationTurn
from forge.swarm.consensus import extract_consensus


def _persona(ap_id: str, archetype: str = "agent") -> AgentPersona:
    return AgentPersona(
        id=ap_id,
        archetype=archetype,
        persona_json=json.dumps({"archetype": archetype}),
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )


def _turn(
    st_id: str,
    agent_id: str,
    *,
    round: int = 3,
    position: str = "support",
    confidence: int = 70,
    reasoning: str = "test reasoning",
    turn_type: str = "reaction",
) -> SimulationTurn:
    content = json.dumps({
        "position": position,
        "confidence": confidence,
        "reasoning": reasoning,
    })
    return SimulationTurn(
        id=st_id,
        simulation_id="s_test",
        round=round,
        agent_persona_id=agent_id,
        turn_type=turn_type,
        content=content,
        position=position,
        confidence=confidence,
        created_at="2026-01-01T00:00:00+00:00",
    )


@pytest.mark.unit
class TestExtractConsensus:
    def test_consensus_majority_position(self):
        """Identifies the position held by most agents in round 3."""
        turns = [
            _turn("st_1", "ap_1", round=3, position="support"),
            _turn("st_2", "ap_2", round=3, position="support"),
            _turn("st_3", "ap_3", round=3, position="oppose"),
        ]
        personas = {f"ap_{i}": _persona(f"ap_{i}") for i in range(1, 4)}
        report = extract_consensus(turns, personas)
        assert report.majority_position == "support"

    def test_consensus_majority_confidence(self):
        """Average confidence of majority agents is computed correctly."""
        turns = [
            _turn("st_1", "ap_1", round=3, position="support", confidence=60),
            _turn("st_2", "ap_2", round=3, position="support", confidence=80),
            _turn("st_3", "ap_3", round=3, position="oppose", confidence=90),
        ]
        personas = {f"ap_{i}": _persona(f"ap_{i}") for i in range(1, 4)}
        report = extract_consensus(turns, personas)
        assert report.majority_confidence == 70.0  # (60+80)/2

    def test_consensus_majority_fraction(self):
        """Fraction of agents in majority is computed correctly."""
        turns = [
            _turn("st_1", "ap_1", round=3, position="support"),
            _turn("st_2", "ap_2", round=3, position="support"),
            _turn("st_3", "ap_3", round=3, position="oppose"),
        ]
        personas = {f"ap_{i}": _persona(f"ap_{i}") for i in range(1, 4)}
        report = extract_consensus(turns, personas)
        assert abs(report.majority_fraction - 2 / 3) < 0.01

    def test_consensus_dissent_clusters(self):
        """Non-majority positions form dissent clusters."""
        turns = [
            _turn("st_1", "ap_1", round=3, position="support"),
            _turn("st_2", "ap_2", round=3, position="support"),
            _turn("st_3", "ap_3", round=3, position="oppose", confidence=80),
            _turn("st_4", "ap_4", round=3, position="oppose", confidence=60),
        ]
        personas = {f"ap_{i}": _persona(f"ap_{i}") for i in range(1, 5)}
        report = extract_consensus(turns, personas)
        assert len(report.dissent_clusters) == 1
        cluster = report.dissent_clusters[0]
        assert cluster.position == "oppose"
        assert cluster.agent_count == 2
        assert cluster.avg_confidence == 70.0

    def test_consensus_conviction_shifts(self):
        """Tracks agents who changed position between round 1 and round 3."""
        turns = [
            # Round 1
            _turn("st_r1_1", "ap_1", round=1, position="oppose", confidence=60),
            _turn("st_r1_2", "ap_2", round=1, position="support", confidence=70),
            # Round 3
            _turn("st_r3_1", "ap_1", round=3, position="support", confidence=80),
            _turn("st_r3_2", "ap_2", round=3, position="support", confidence=75),
        ]
        personas = {
            "ap_1": _persona("ap_1", "shifter"),
            "ap_2": _persona("ap_2", "steady"),
        }
        report = extract_consensus(turns, personas)
        assert len(report.conviction_shifts) == 1
        shift = report.conviction_shifts[0]
        assert shift.agent_persona_id == "ap_1"
        assert shift.from_position == "oppose"
        assert shift.to_position == "support"
        assert shift.confidence_delta == 20  # 80 - 60

    def test_consensus_no_shifts(self):
        """No conviction shifts when all agents maintain position."""
        turns = [
            _turn("st_r1_1", "ap_1", round=1, position="support", confidence=70),
            _turn("st_r1_2", "ap_2", round=1, position="oppose", confidence=60),
            _turn("st_r3_1", "ap_1", round=3, position="support", confidence=75),
            _turn("st_r3_2", "ap_2", round=3, position="oppose", confidence=65),
        ]
        personas = {"ap_1": _persona("ap_1"), "ap_2": _persona("ap_2")}
        report = extract_consensus(turns, personas)
        assert len(report.conviction_shifts) == 0

    def test_consensus_edge_cases_unique_positions(self):
        """Positions held by 1-2 agents are flagged as edge cases."""
        turns = [
            _turn("st_1", "ap_1", round=3, position="support"),
            _turn("st_2", "ap_2", round=3, position="support"),
            _turn("st_3", "ap_3", round=3, position="support"),
            _turn("st_4", "ap_4", round=3, position="conditional",
                  reasoning="Only if inflation stays low"),
        ]
        personas = {f"ap_{i}": _persona(f"ap_{i}") for i in range(1, 5)}
        report = extract_consensus(turns, personas)
        assert len(report.edge_cases) >= 1
        assert any(ec.agent_persona_id == "ap_4" for ec in report.edge_cases)

    def test_consensus_unanimous(self):
        """All agents agree — no dissent clusters."""
        turns = [
            _turn("st_1", "ap_1", round=3, position="support", confidence=80),
            _turn("st_2", "ap_2", round=3, position="support", confidence=70),
            _turn("st_3", "ap_3", round=3, position="support", confidence=90),
        ]
        personas = {f"ap_{i}": _persona(f"ap_{i}") for i in range(1, 4)}
        report = extract_consensus(turns, personas)
        assert report.majority_position == "support"
        assert report.majority_fraction == 1.0
        assert report.dissent_clusters == []

    def test_consensus_empty_turns(self):
        """Empty turns returns sensible defaults."""
        report = extract_consensus([], {})
        assert report.majority_position == "neutral"
        assert report.majority_confidence == 0.0
        assert report.majority_fraction == 0.0
        assert report.dissent_clusters == []
        assert report.conviction_shifts == []
        assert report.edge_cases == []

    def test_consensus_tie_highest_confidence_wins(self):
        """Tie broken by highest average confidence."""
        turns = [
            _turn("st_1", "ap_1", round=3, position="support", confidence=90),
            _turn("st_2", "ap_2", round=3, position="support", confidence=80),
            _turn("st_3", "ap_3", round=3, position="oppose", confidence=60),
            _turn("st_4", "ap_4", round=3, position="oppose", confidence=50),
        ]
        personas = {f"ap_{i}": _persona(f"ap_{i}") for i in range(1, 5)}
        report = extract_consensus(turns, personas)
        # 2 vs 2, but support has higher avg confidence (85 vs 55)
        assert report.majority_position == "support"

    def test_consensus_uses_only_round3_for_majority(self):
        """Majority is computed from round 3 turns only."""
        turns = [
            # Round 1 — different positions
            _turn("st_r1_1", "ap_1", round=1, position="oppose"),
            _turn("st_r1_2", "ap_2", round=1, position="oppose"),
            # Round 3 — both shifted to support
            _turn("st_r3_1", "ap_1", round=3, position="support"),
            _turn("st_r3_2", "ap_2", round=3, position="support"),
        ]
        personas = {"ap_1": _persona("ap_1"), "ap_2": _persona("ap_2")}
        report = extract_consensus(turns, personas)
        assert report.majority_position == "support"


@pytest.mark.unit
class TestConfidenceTrend:
    def test_confidence_trend_per_round(self):
        """confidence_trend has one avg confidence value per round."""
        turns = [
            _turn("s1", "a1", round=1, confidence=60),
            _turn("s2", "a2", round=1, confidence=80),
            _turn("s3", "a1", round=2, confidence=70),
            _turn("s4", "a2", round=2, confidence=90),
            _turn("s5", "a1", round=3, confidence=75),
            _turn("s6", "a2", round=3, confidence=85),
        ]
        personas = {"a1": _persona("a1"), "a2": _persona("a2")}
        report = extract_consensus(turns, personas)
        assert len(report.confidence_trend) == 3
        assert report.confidence_trend[0] == 70.0   # avg(60,80)
        assert report.confidence_trend[1] == 80.0   # avg(70,90)
        assert report.confidence_trend[2] == 80.0   # avg(75,85)

    def test_confidence_trend_empty_turns(self):
        """Empty turns produce empty trend."""
        report = extract_consensus([], {})
        assert report.confidence_trend == []

    def test_confidence_trend_single_round(self):
        """Only round 1 turns produce single-element trend."""
        turns = [
            _turn("s1", "a1", round=1, confidence=50),
        ]
        personas = {"a1": _persona("a1")}
        report = extract_consensus(turns, personas)
        assert report.confidence_trend == [50.0]
