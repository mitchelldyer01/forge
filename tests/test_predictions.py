"""
Tests for forge.swarm.predictions — prediction extraction from simulation output.

Mirrors: forge/swarm/predictions.py
"""

import pytest

from forge.db.models import Prediction
from forge.db.store import Store
from forge.swarm.consensus import ConsensusReport, DissentCluster
from forge.swarm.population import SeedMaterial
from forge.swarm.predictions import extract_predictions, format_consensus_for_prompt


@pytest.mark.unit
class TestFormatConsensusForPrompt:
    def test_format_basic_consensus(self):
        """Formats a consensus report into readable text."""
        report = ConsensusReport(
            majority_position="support",
            majority_confidence=75.0,
            majority_fraction=0.8,
            dissent_clusters=[
                DissentCluster(
                    position="oppose",
                    agent_count=2,
                    avg_confidence=60.0,
                    key_arguments=["Risk is too high"],
                ),
            ],
        )
        text = format_consensus_for_prompt(report)
        assert "support" in text
        assert "75" in text
        assert "80%" in text or "0.8" in text
        assert "oppose" in text
        assert "Risk is too high" in text

    def test_format_empty_consensus(self):
        """Formats an empty consensus report without crashing."""
        report = ConsensusReport(
            majority_position="neutral",
            majority_confidence=0.0,
            majority_fraction=0.0,
        )
        text = format_consensus_for_prompt(report)
        assert "neutral" in text


@pytest.mark.unit
class TestExtractPredictions:
    @pytest.fixture
    def seed(self) -> SeedMaterial:
        return SeedMaterial(text="EU regulates AI agents")

    @pytest.fixture
    def consensus(self) -> ConsensusReport:
        return ConsensusReport(
            majority_position="support",
            majority_confidence=75.0,
            majority_fraction=0.7,
        )

    @pytest.fixture
    def simulation(self, db: Store):
        return db.save_simulation(mode="scenario", seed_text="EU regulates AI agents")

    async def test_extract_predictions_returns_list(
        self, seed, consensus, simulation, mock_llm, db: Store,
    ):
        """extract_predictions returns a list of Prediction models."""
        mock_llm.set_response({
            "predictions": [
                {
                    "claim": "EU will pass AI agent regulation by Q3 2027",
                    "confidence": 70,
                    "consensus_strength": 0.8,
                    "resolution_deadline": "2027-09-30T00:00:00+00:00",
                    "dissent_summary": "Some argued self-regulation is more likely",
                    "tags": ["regulation", "ai"],
                },
                {
                    "claim": "AI agent startups will relocate to non-EU jurisdictions",
                    "confidence": 55,
                    "consensus_strength": 0.5,
                    "resolution_deadline": "2028-01-01T00:00:00+00:00",
                    "dissent_summary": "EU market too large to abandon",
                    "tags": ["regulation", "business"],
                },
            ],
        })
        predictions = await extract_predictions(
            seed, consensus, mock_llm, db, simulation.id,
        )
        assert len(predictions) == 2
        assert all(isinstance(p, Prediction) for p in predictions)

    async def test_extract_predictions_persists_to_store(
        self, seed, consensus, simulation, mock_llm, db: Store,
    ):
        """Predictions are persisted to the database."""
        mock_llm.set_response({
            "predictions": [
                {
                    "claim": "Prediction A",
                    "confidence": 65,
                    "consensus_strength": 0.7,
                },
            ],
        })
        predictions = await extract_predictions(
            seed, consensus, mock_llm, db, simulation.id,
        )
        stored = db.list_predictions(simulation_id=simulation.id)
        assert len(stored) == 1
        assert stored[0].id == predictions[0].id

    async def test_extract_predictions_updates_simulation_count(
        self, seed, consensus, simulation, mock_llm, db: Store,
    ):
        """simulation.predictions_extracted is updated."""
        mock_llm.set_response({
            "predictions": [
                {"claim": "P1", "confidence": 60},
                {"claim": "P2", "confidence": 70},
            ],
        })
        await extract_predictions(seed, consensus, mock_llm, db, simulation.id)
        sim = db.get_simulation(simulation.id)
        assert sim is not None
        assert sim.predictions_extracted == 2

    async def test_extract_predictions_empty_output(
        self, seed, consensus, simulation, mock_llm, db: Store,
    ):
        """Empty predictions from LLM returns empty list."""
        mock_llm.set_response({"predictions": []})
        predictions = await extract_predictions(
            seed, consensus, mock_llm, db, simulation.id,
        )
        assert predictions == []

    async def test_extract_predictions_skips_missing_claim(
        self, seed, consensus, simulation, mock_llm, db: Store,
    ):
        """Predictions missing a claim are skipped."""
        mock_llm.set_response({
            "predictions": [
                {"confidence": 60},  # missing claim
                {"claim": "Valid prediction", "confidence": 70},
            ],
        })
        predictions = await extract_predictions(
            seed, consensus, mock_llm, db, simulation.id,
        )
        assert len(predictions) == 1
        assert predictions[0].claim == "Valid prediction"

    async def test_extract_predictions_uses_prompt_with_seed(
        self, seed, consensus, simulation, mock_llm, db: Store,
    ):
        """The LLM is called with prompt containing seed text."""
        mock_llm.set_response({"predictions": []})
        await extract_predictions(seed, consensus, mock_llm, db, simulation.id)
        assert mock_llm.call_count == 1
        last_msg = mock_llm.last_messages[-1]["content"]
        assert "EU regulates AI agents" in last_msg
