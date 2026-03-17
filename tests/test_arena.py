"""
Tests for forge.swarm.arena — multi-round simulation loop.

Mirrors: forge/swarm/arena.py
"""

import json

import httpx
import pytest

from forge.db.models import AgentPersona, Simulation
from forge.db.store import Store
from forge.llm.client import CompletionResponse, MockLLMClient, ParseError
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


@pytest.mark.unit
class TestArenaAgentResilience:
    """Tests that arena rounds handle per-agent failures gracefully."""

    @pytest.fixture
    def seed(self) -> SeedMaterial:
        return SeedMaterial(text="Test scenario for resilience")

    @pytest.fixture
    def three_agents(self, db: Store) -> list[AgentPersona]:
        return [
            _make_persona(db, "optimist", "Alice"),
            _make_persona(db, "pessimist", "Bob"),
            _make_persona(db, "analyst", "Carol"),
        ]

    async def test_arena_round1_continues_when_one_agent_fails(
        self, seed: SeedMaterial, three_agents, db: Store,
    ):
        """One agent raises ParseError in round 1, others succeed."""
        call_counter = 0

        async def failing_complete(messages, **kwargs):
            nonlocal call_counter
            call_counter += 1
            if call_counter == 2:  # second agent fails
                raise ParseError("truncated", ValueError("bad json"))
            data = _round1_response()
            return CompletionResponse(
                content=json.dumps(data),
                parsed_json=data,
                token_count=50,
                raw_response={"mock": True},
            )

        mock = MockLLMClient()
        mock.complete = failing_complete

        # Need round 2 and 3 to also work — queue enough for remaining agents
        # But first let's just run 1 round
        result = await run_simulation(
            seed, three_agents, mock, db, rounds=1, max_concurrent=1,
        )
        assert result.simulation.status == "complete"
        round1_turns = db.list_turns_by_simulation(result.simulation.id, round=1)
        # 2 agents succeeded, 1 failed and was skipped
        assert len(round1_turns) == 2

    async def test_arena_round1_timeout_skips_agent(
        self, seed: SeedMaterial, three_agents, db: Store,
    ):
        """One agent times out in round 1, others succeed."""
        call_counter = 0

        async def timeout_complete(messages, **kwargs):
            nonlocal call_counter
            call_counter += 1
            if call_counter == 1:  # first agent times out
                raise httpx.ReadTimeout("read timed out")
            data = _round1_response()
            return CompletionResponse(
                content=json.dumps(data),
                parsed_json=data,
                token_count=50,
                raw_response={"mock": True},
            )

        mock = MockLLMClient()
        mock.complete = timeout_complete

        result = await run_simulation(
            seed, three_agents, mock, db, rounds=1, max_concurrent=1,
        )
        assert result.simulation.status == "complete"
        round1_turns = db.list_turns_by_simulation(result.simulation.id, round=1)
        assert len(round1_turns) == 2


@pytest.mark.unit
class TestArenaOnTurnCallback:
    """Tests for the on_turn real-time callback."""

    @pytest.fixture
    def seed(self) -> SeedMaterial:
        return SeedMaterial(text="Test scenario for callback")

    @pytest.fixture
    def two_agents(self, db: Store) -> list[AgentPersona]:
        return [
            _make_persona(db, "optimist", "Alice"),
            _make_persona(db, "pessimist", "Bob"),
        ]

    def _queue_responses_for_agents(self, mock_llm, agent_count: int):
        responses = []
        for _ in range(agent_count):
            responses.append(_round1_response())
        for _ in range(agent_count):
            responses.append(_round2_response())
        for _ in range(agent_count):
            responses.append(_round3_response())
        mock_llm.set_responses(responses)

    async def test_on_turn_called_for_each_successful_turn(
        self, seed: SeedMaterial, two_agents, mock_llm, db: Store,
    ):
        """on_turn is called once per successful agent turn."""
        self._queue_responses_for_agents(mock_llm, 2)
        received = []

        def on_turn(turn, round_num, agent):
            received.append((round_num, agent.archetype))

        await run_simulation(
            seed, two_agents, mock_llm, db,
            rounds=3, max_concurrent=1, on_turn=on_turn,
        )
        # 2 agents * 3 rounds = 6 callbacks
        assert len(received) == 6
        # Round 1 comes first
        assert all(r == 1 for r, _ in received[:2])
        assert all(r == 2 for r, _ in received[2:4])
        assert all(r == 3 for r, _ in received[4:6])

    async def test_on_turn_not_called_for_failed_agents(
        self, seed: SeedMaterial, db: Store,
    ):
        """on_turn is NOT called for agents that fail."""
        agents = [
            _make_persona(db, "optimist", "Alice"),
            _make_persona(db, "pessimist", "Bob"),
            _make_persona(db, "analyst", "Carol"),
        ]
        call_counter = 0

        async def failing_complete(messages, **kwargs):
            nonlocal call_counter
            call_counter += 1
            if call_counter == 2:
                raise ParseError("truncated", ValueError("bad"))
            data = _round1_response()
            return CompletionResponse(
                content=json.dumps(data),
                parsed_json=data,
                token_count=50,
                raw_response={"mock": True},
            )

        mock = MockLLMClient()
        mock.complete = failing_complete

        received = []

        def on_turn(turn, round_num, agent):
            received.append(agent.archetype)

        await run_simulation(
            seed, agents, mock, db,
            rounds=1, max_concurrent=1, on_turn=on_turn,
        )
        # 2 succeeded, 1 failed → 2 callbacks
        assert len(received) == 2

    async def test_simulation_works_without_on_turn(
        self, seed: SeedMaterial, two_agents, mock_llm, db: Store,
    ):
        """on_turn is optional — simulation works without it."""
        self._queue_responses_for_agents(mock_llm, 2)
        result = await run_simulation(
            seed, two_agents, mock_llm, db, rounds=3, max_concurrent=1,
        )
        assert result.simulation.status == "complete"
