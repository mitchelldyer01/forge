"""Tests for headless simulation orchestration with prior knowledge injection.

Mirrors: forge/swarm/orchestrate.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from forge.db.store import Store
    from forge.llm.client import MockLLMClient


def _agent_response() -> dict:
    return {"agents": [{"archetype": "analyst", "name": "Alice"}]}


def _turn_response() -> dict:
    return {
        "position": "support",
        "confidence": 75,
        "justification": "Test reasoning",
        "evidence": ["test evidence"],
        "delta": 0,
    }


def _prediction_response() -> dict:
    return {"predictions": [
        {
            "claim": "Test prediction",
            "confidence": 70,
            "consensus_strength": 0.8,
            "resolution_deadline": "2027-01-01",
        },
    ]}


def _queue_full_simulation(mock_llm: MockLLMClient) -> None:
    """Queue responses for a complete 1-agent, 3-round simulation."""
    mock_llm.set_responses([
        _agent_response(),
        _turn_response(),
        _turn_response(),
        _turn_response(),
        _prediction_response(),
    ])


@pytest.mark.unit
class TestPriorKnowledgeInjection:
    def test_simulation_injects_prior_knowledge(
        self, db: Store, mock_llm: MockLLMClient,
    ) -> None:
        """When related hypotheses exist with embeddings, they appear in seed context."""
        from forge.swarm.orchestrate import inject_prior_knowledge
        from forge.swarm.population import SeedMaterial

        # Create a hypothesis with embedding
        h = db.save_hypothesis(
            claim="EU AI regulation will reduce innovation",
            source="manual",
            confidence=80,
        )
        embedding = np.random.default_rng(42).random(384).astype(np.float32)
        db.update_hypothesis(h.id, embedding=embedding.tobytes())

        seed = SeedMaterial(text="Will EU AI Act hurt startups?")
        # Use a mock embed function that returns a similar vector
        enriched = inject_prior_knowledge(
            seed, db, _embed_fn=lambda _text: embedding,
        )
        assert enriched.context is not None
        assert "EU AI regulation will reduce innovation" in enriched.context

    def test_simulation_works_without_prior_knowledge(
        self, db: Store, mock_llm: MockLLMClient,
    ) -> None:
        """No related hypotheses → seed unchanged."""
        from forge.swarm.orchestrate import inject_prior_knowledge
        from forge.swarm.population import SeedMaterial

        seed = SeedMaterial(text="Completely novel topic")
        enriched = inject_prior_knowledge(
            seed, db, _embed_fn=lambda _text: np.zeros(384, dtype=np.float32),
        )
        # Context should be empty or None (no prior knowledge found)
        assert enriched.context is None or "confidence" not in enriched.context

    def test_prior_knowledge_limited_to_top_5(
        self, db: Store, mock_llm: MockLLMClient,
    ) -> None:
        """At most 5 prior hypotheses are injected."""
        from forge.swarm.orchestrate import inject_prior_knowledge
        from forge.swarm.population import SeedMaterial

        embedding = np.random.default_rng(42).random(384).astype(np.float32)
        for i in range(10):
            h = db.save_hypothesis(
                claim=f"Hypothesis {i} about AI regulation",
                source="manual",
                confidence=60 + i,
            )
            # Slightly different embeddings
            vec = embedding + np.float32(i * 0.01)
            db.update_hypothesis(h.id, embedding=vec.tobytes())

        seed = SeedMaterial(text="AI regulation impact")
        enriched = inject_prior_knowledge(
            seed, db, _embed_fn=lambda _text: embedding, limit=5,
        )
        # Count how many hypotheses appear in context
        assert enriched.context is not None
        hypothesis_count = enriched.context.count("confidence:")
        assert hypothesis_count <= 5

    async def test_orchestrate_with_prior_knowledge(
        self, db: Store, mock_llm: MockLLMClient,
    ) -> None:
        """orchestrate_simulation injects prior knowledge when available."""
        from forge.swarm.orchestrate import orchestrate_simulation
        from forge.swarm.population import SeedMaterial

        # Create a hypothesis with embedding
        h = db.save_hypothesis(
            claim="Prior knowledge about AI",
            source="manual", confidence=80,
        )
        embedding = np.random.default_rng(42).random(384).astype(np.float32)
        db.update_hypothesis(h.id, embedding=embedding.tobytes())

        _queue_full_simulation(mock_llm)

        seed = SeedMaterial(text="AI regulation question")
        result = await orchestrate_simulation(
            seed, mock_llm, db,
            agent_count=1, rounds=3,
            _embed_fn=lambda _text: embedding,
        )
        assert result.simulation.status == "complete"
