"""
Shared test fixtures for FORGE.

All tests use MockLLMClient. No test ever hits a real LLM endpoint.
DB fixtures provide a fresh in-memory SQLite instance per test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from forge.db.store import Store
from forge.llm.client import MockLLMClient

if TYPE_CHECKING:
    from forge.db.models import (
        AgentPersona,
        Article,
        Evidence,
        Feed,
        Hypothesis,
        Prediction,
        Simulation,
    )
    from forge.swarm.population import SeedMaterial


@pytest.fixture
def db() -> Store:
    """Fresh in-memory Store with schema applied, torn down after test."""
    return Store(":memory:")


@pytest.fixture
def mock_llm() -> MockLLMClient:
    """MockLLMClient that returns configurable JSON responses."""
    return MockLLMClient()


@pytest.fixture
def sample_hypothesis(db: Store) -> Hypothesis:
    """A pre-built Hypothesis model for reuse."""
    return db.save_hypothesis(
        claim="AI agents will displace 30% of SaaS by 2027",
        source="manual",
        context="Based on current trends in AI agent capabilities",
        confidence=65,
        tags=["ai", "saas", "prediction"],
    )


@pytest.fixture
def sample_evidence(db: Store) -> Evidence:
    """A pre-built Evidence model for reuse."""
    return db.save_evidence(
        content="OpenAI reported 200M weekly active ChatGPT users in Aug 2024",
        source_url="https://example.com/openai-users",
        source_name="TechCrunch",
    )


@pytest.fixture
def sample_simulation(db: Store) -> Simulation:
    """A pre-built Simulation model for reuse."""
    return db.save_simulation(
        mode="scenario",
        seed_text="What if the EU regulates AI agents?",
        agent_count=5,
        rounds=3,
    )


@pytest.fixture
def sample_agent_persona(db: Store) -> AgentPersona:
    """A pre-built AgentPersona model for reuse."""
    import json

    persona = {
        "archetype": "tech_optimist",
        "name": "Alice Chen",
        "background": "Former Google engineer turned AI startup founder",
        "expertise": ["machine_learning", "product_strategy"],
        "personality": {
            "risk_appetite": "high",
            "optimism_bias": "optimist",
            "contrarian_tendency": 0.3,
            "analytical_depth": "deep",
        },
        "initial_stance": "AI regulation will slow innovation",
        "reasoning_style": "data-driven, first-principles",
    }
    return db.save_agent_persona(
        archetype="tech_optimist",
        persona_json=json.dumps(persona),
    )


@pytest.fixture
def sample_seed() -> SeedMaterial:
    """A pre-built SeedMaterial for reuse."""
    from forge.swarm.population import SeedMaterial

    return SeedMaterial(
        text="The EU announces strict AI agent regulations",
        context="European Commission proposal for AI Act extension",
    )


@pytest.fixture
def sample_feed(db: Store) -> Feed:
    """A pre-built Feed model for reuse."""
    return db.save_feed(
        name="TechCrunch",
        url="https://techcrunch.com/feed/",
        poll_interval_minutes=240,
    )


@pytest.fixture
def sample_article(db: Store, sample_feed: Feed) -> Article:
    """A pre-built Article model for reuse."""
    return db.save_article(
        url="https://techcrunch.com/2024/01/15/ai-agents",
        feed_id=sample_feed.id,
        title="AI Agents Are Reshaping Enterprise Software",
        content="The rise of AI agents is fundamentally changing how enterprises build software.",
    )


@pytest.fixture
def sample_prediction(db: Store, sample_simulation: Simulation) -> Prediction:
    """A pre-built Prediction model with resolution deadline for reuse."""
    return db.save_prediction(
        simulation_id=sample_simulation.id,
        claim="AI agent adoption will exceed 50% in enterprise by 2026",
        confidence=72,
        consensus_strength=0.8,
        dissent_summary="Some argue enterprises are too slow to adopt",
        resolution_deadline="2026-12-31T00:00:00+00:00",
    )
