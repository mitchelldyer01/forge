"""Tests for auto-simulation pipeline: claim selection and orchestration.

Mirrors: forge/pipeline/auto_simulate.py, forge/swarm/orchestrate.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from forge.db.store import Store
    from forge.llm.client import MockLLMClient


def _make_hypothesis(db: Store, claim: str, confidence: int, source: str = "rss") -> str:
    """Create a hypothesis and return its ID."""
    h = db.save_hypothesis(
        claim=claim, source=source, confidence=confidence, tags=["test"],
    )
    return h.id


# ------------------------------------------------------------------
# Claim selection for auto-simulation
# ------------------------------------------------------------------


@pytest.mark.unit
class TestSelectClaimsForSimulation:
    def test_selects_top_n_by_confidence(self, db: Store) -> None:
        """Picks the top N claims sorted by confidence descending."""
        from forge.pipeline.auto_simulate import select_claims_for_simulation

        _make_hypothesis(db, "Low confidence", 40, source="rss")
        _make_hypothesis(db, "High confidence", 90, source="rss")
        _make_hypothesis(db, "Medium confidence", 70, source="rss")

        selected = select_claims_for_simulation(db, top_n=2, min_confidence=50)
        assert len(selected) == 2
        assert selected[0].confidence >= selected[1].confidence

    def test_filters_below_min_confidence(self, db: Store) -> None:
        """Claims below min_confidence are excluded."""
        from forge.pipeline.auto_simulate import select_claims_for_simulation

        _make_hypothesis(db, "Low", 30, source="rss")
        _make_hypothesis(db, "High", 80, source="rss")

        selected = select_claims_for_simulation(db, top_n=5, min_confidence=60)
        assert len(selected) == 1
        assert selected[0].confidence == 80

    def test_respects_top_n_limit(self, db: Store) -> None:
        """Never returns more than top_n claims."""
        from forge.pipeline.auto_simulate import select_claims_for_simulation

        for i in range(10):
            _make_hypothesis(db, f"Claim {i}", 70 + i, source="rss")

        selected = select_claims_for_simulation(db, top_n=3, min_confidence=60)
        assert len(selected) <= 3

    def test_empty_when_no_claims(self, db: Store) -> None:
        """Returns empty list when no hypotheses exist."""
        from forge.pipeline.auto_simulate import select_claims_for_simulation

        selected = select_claims_for_simulation(db, top_n=3, min_confidence=60)
        assert selected == []

    def test_excludes_non_alive_hypotheses(self, db: Store) -> None:
        """Only 'alive' hypotheses are candidates."""
        from forge.pipeline.auto_simulate import select_claims_for_simulation

        h_id = _make_hypothesis(db, "Dead claim", 80, source="rss")
        db.update_hypothesis(h_id, status="dead")
        _make_hypothesis(db, "Alive claim", 75, source="rss")

        selected = select_claims_for_simulation(db, top_n=5, min_confidence=60)
        assert len(selected) == 1
        assert selected[0].claim == "Alive claim"


# ------------------------------------------------------------------
# Simulation orchestration (headless, no Rich UI)
# ------------------------------------------------------------------


@pytest.mark.unit
class TestOrchestrateSimulation:
    async def test_orchestrate_returns_result(
        self, db: Store, mock_llm: MockLLMClient,
    ) -> None:
        """orchestrate_simulation runs population → debate → predictions."""
        from forge.swarm.orchestrate import orchestrate_simulation
        from forge.swarm.population import SeedMaterial

        # Queue responses: population batch + round1 + round2 + round3 + predictions
        agent_data = {"agents": [
            {"archetype": "analyst", "name": "Alice"},
        ]}
        turn_data = {
            "position": "support",
            "confidence": 75,
            "justification": "Test reasoning",
            "evidence": ["test evidence"],
            "delta": 0,
        }
        prediction_data = {"predictions": [
            {
                "claim": "Test prediction",
                "confidence": 70,
                "consensus_strength": 0.8,
                "resolution_deadline": "2027-01-01",
            },
        ]}
        mock_llm.set_responses([
            agent_data,       # population generation
            turn_data,        # round 1
            turn_data,        # round 2
            turn_data,        # round 3
            prediction_data,  # prediction extraction
        ])

        seed = SeedMaterial(text="Test scenario", context="Test context")
        result = await orchestrate_simulation(seed, mock_llm, db, agent_count=1, rounds=3)

        assert result.simulation.status == "complete"
        assert len(result.predictions) >= 1
        assert result.predictions[0].claim == "Test prediction"

    async def test_orchestrate_no_agents_raises(
        self, db: Store, mock_llm: MockLLMClient,
    ) -> None:
        """Raises RuntimeError when no agents can be generated."""
        from forge.swarm.orchestrate import orchestrate_simulation
        from forge.swarm.population import SeedMaterial

        mock_llm.set_response({"agents": []})

        seed = SeedMaterial(text="Test scenario")
        with pytest.raises(RuntimeError, match="No agents generated"):
            await orchestrate_simulation(seed, mock_llm, db, agent_count=5)


# ------------------------------------------------------------------
# Auto-simulate cycle (integration of selection + orchestration)
# ------------------------------------------------------------------


@pytest.mark.unit
class TestAutoSimulateCycle:
    async def test_noop_when_no_claims(
        self, db: Store, mock_llm: MockLLMClient,
    ) -> None:
        """Empty DB means no simulations triggered."""
        from forge.pipeline.auto_simulate import auto_simulate_cycle

        result = await auto_simulate_cycle(db, mock_llm, top_n=3, min_confidence=60)
        assert result == []

    async def test_runs_simulation_for_selected_claims(
        self, db: Store, mock_llm: MockLLMClient,
    ) -> None:
        """Selected claims get simulated and return simulation IDs."""
        from forge.pipeline.auto_simulate import auto_simulate_cycle

        _make_hypothesis(db, "EU AI regulation will reduce VC funding", 80, source="rss")

        agent_data = {"agents": [
            {"archetype": "analyst", "name": "Alice"},
        ]}
        turn_data = {
            "position": "support",
            "confidence": 75,
            "justification": "Test reasoning",
            "evidence": ["test evidence"],
            "delta": 0,
        }
        prediction_data = {"predictions": [
            {
                "claim": "VC funding will drop 20%",
                "confidence": 65,
                "consensus_strength": 0.7,
                "resolution_deadline": "2027-06-01",
            },
        ]}
        mock_llm.set_responses([
            agent_data, turn_data, turn_data, turn_data, prediction_data,
        ])

        sim_ids = await auto_simulate_cycle(
            db, mock_llm, top_n=1, min_confidence=60, agent_count=1, rounds=3,
        )
        assert len(sim_ids) == 1
        # Verify simulation was persisted
        sim = db.get_simulation(sim_ids[0])
        assert sim is not None
        assert sim.status == "complete"
