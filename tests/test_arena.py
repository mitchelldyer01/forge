"""
Tests for forge.swarm.arena — multi-round simulation loop.

Mirrors: forge/swarm/arena.py
"""

import json

import pytest

from forge.db.models import AgentPersona, Simulation
from forge.db.store import Store
from forge.swarm.arena import SimulationResult, run_simulation
from forge.swarm.consensus import ConsensusReport
from forge.swarm.population import SeedMaterial


def _make_persona(db: Store, archetype: str, name: str = "Agent") -> AgentPersona:
    """Create and persist an agent persona."""
    persona_data = {
        "archetype": archetype,
        "name": name,
        "background": f"{archetype} background",
        "expertise": ["testing"],
        "personality": {
            "risk_appetite": "medium",
            "optimism_bias": "realist",
            "contrarian_tendency": 0.3,
            "analytical_depth": "deep",
        },
        "initial_stance": "Test stance",
        "reasoning_style": "analytical",
    }
    return db.save_agent_persona(
        archetype=archetype,
        persona_json=json.dumps(persona_data),
    )


def _round1_response() -> dict:
    return {
        "position": "support",
        "confidence": 70,
        "reasoning": "Test reasoning for round 1",
        "key_concern": "Test concern",
    }


def _round2_response() -> dict:
    return {
        "turn_type": "challenge",
        "position": "support",
        "confidence": 75,
        "reasoning": "Responding to opposing view",
        "key_point": "Test key point",
    }


def _round3_response() -> dict:
    return {
        "final_position": "support",
        "confidence": 80,
        "conviction_delta": 10,
        "changed_mind": False,
        "reasoning": "Final synthesis",
        "key_insight": "Test insight",
    }


@pytest.mark.unit
class TestRunSimulation:
    @pytest.fixture
    def seed(self) -> SeedMaterial:
        return SeedMaterial(text="Test scenario for simulation")

    @pytest.fixture
    def two_agents(self, db: Store) -> list[AgentPersona]:
        return [
            _make_persona(db, "optimist", "Alice"),
            _make_persona(db, "pessimist", "Bob"),
        ]

    def _queue_responses_for_agents(self, mock_llm, agent_count: int, rounds: int):
        """Queue enough mock responses for a full simulation."""
        responses = []
        for _ in range(agent_count):
            responses.append(_round1_response())
        for _ in range(agent_count):
            responses.append(_round2_response())
        for _ in range(agent_count):
            responses.append(_round3_response())
        mock_llm.set_responses(responses)

    async def test_arena_creates_simulation_record(
        self, seed: SeedMaterial, two_agents, mock_llm, db: Store,
    ):
        """run_simulation creates a simulation record in the store."""
        self._queue_responses_for_agents(mock_llm, 2, 3)
        result = await run_simulation(
            seed, two_agents, mock_llm, db, rounds=3, max_concurrent=1,
        )
        assert isinstance(result, SimulationResult)
        assert isinstance(result.simulation, Simulation)
        assert result.simulation.status == "complete"
        assert result.simulation.mode == "scenario"

    async def test_arena_round1_produces_turns(
        self, seed: SeedMaterial, two_agents, mock_llm, db: Store,
    ):
        """Round 1 creates one turn per agent."""
        self._queue_responses_for_agents(mock_llm, 2, 3)
        result = await run_simulation(
            seed, two_agents, mock_llm, db, rounds=3, max_concurrent=1,
        )
        round1_turns = db.list_turns_by_simulation(result.simulation.id, round=1)
        assert len(round1_turns) == 2
        assert all(t.turn_type == "reaction" for t in round1_turns)

    async def test_arena_three_rounds_produces_all_turns(
        self, seed: SeedMaterial, two_agents, mock_llm, db: Store,
    ):
        """Full 3-round simulation produces 2 turns per round = 6 total."""
        self._queue_responses_for_agents(mock_llm, 2, 3)
        result = await run_simulation(
            seed, two_agents, mock_llm, db, rounds=3, max_concurrent=1,
        )
        all_turns = db.list_turns_by_simulation(result.simulation.id)
        assert len(all_turns) == 6  # 2 agents * 3 rounds

    async def test_arena_records_duration(
        self, seed: SeedMaterial, two_agents, mock_llm, db: Store,
    ):
        """Simulation records a positive duration."""
        self._queue_responses_for_agents(mock_llm, 2, 3)
        result = await run_simulation(
            seed, two_agents, mock_llm, db, rounds=3, max_concurrent=1,
        )
        assert result.duration_seconds > 0
        sim = db.get_simulation(result.simulation.id)
        assert sim is not None
        assert sim.duration_seconds is not None
        assert sim.duration_seconds > 0

    async def test_arena_returns_consensus(
        self, seed: SeedMaterial, two_agents, mock_llm, db: Store,
    ):
        """Simulation result includes a ConsensusReport."""
        self._queue_responses_for_agents(mock_llm, 2, 3)
        result = await run_simulation(
            seed, two_agents, mock_llm, db, rounds=3, max_concurrent=1,
        )
        assert isinstance(result.consensus, ConsensusReport)

    async def test_arena_saves_turn_metadata(
        self, seed: SeedMaterial, two_agents, mock_llm, db: Store,
    ):
        """Turns have position and confidence populated from LLM output."""
        self._queue_responses_for_agents(mock_llm, 2, 3)
        result = await run_simulation(
            seed, two_agents, mock_llm, db, rounds=3, max_concurrent=1,
        )
        round1_turns = db.list_turns_by_simulation(result.simulation.id, round=1)
        for turn in round1_turns:
            assert turn.position is not None
            assert turn.confidence is not None

    async def test_arena_simulation_persisted_with_agent_count(
        self, seed: SeedMaterial, two_agents, mock_llm, db: Store,
    ):
        """Simulation record has correct agent_count and rounds."""
        self._queue_responses_for_agents(mock_llm, 2, 3)
        result = await run_simulation(
            seed, two_agents, mock_llm, db, rounds=3, max_concurrent=1,
        )
        sim = db.get_simulation(result.simulation.id)
        assert sim is not None
        assert sim.agent_count == 2
        assert sim.rounds == 3
