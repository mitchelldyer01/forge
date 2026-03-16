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
    from forge.db.models import Evidence, Hypothesis


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
