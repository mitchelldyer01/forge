"""
Tests for forge.swarm.population — agent persona generation from seed material.

Mirrors: forge/swarm/population.py
"""

import pytest

from forge.db.models import AgentPersona
from forge.db.store import Store
from forge.swarm.population import SeedMaterial, generate_population


@pytest.mark.unit
class TestSeedMaterial:
    def test_seed_material_creation(self):
        """SeedMaterial can be created with text and optional context."""
        seed = SeedMaterial(text="Test scenario")
        assert seed.text == "Test scenario"
        assert seed.context is None

    def test_seed_material_with_context(self):
        """SeedMaterial can be created with context."""
        seed = SeedMaterial(text="Test", context="Background info")
        assert seed.context == "Background info"


@pytest.mark.unit
class TestGeneratePopulation:
    @pytest.fixture
    def seed(self) -> SeedMaterial:
        return SeedMaterial(
            text="The EU announces strict AI agent regulations",
            context="European Commission proposal",
        )

    async def test_generate_population_returns_agent_personas(
        self, seed: SeedMaterial, mock_llm, db: Store,
    ):
        """generate_population returns a list of AgentPersona models."""
        mock_llm.set_response({
            "agents": [
                {
                    "archetype": "tech_optimist",
                    "name": "Alice Chen",
                    "background": "AI researcher",
                    "expertise": ["machine_learning"],
                    "personality": {
                        "risk_appetite": "high",
                        "optimism_bias": "optimist",
                        "contrarian_tendency": 0.3,
                        "analytical_depth": "deep",
                    },
                    "initial_stance": "Regulation will slow innovation",
                    "reasoning_style": "data-driven",
                },
                {
                    "archetype": "policy_analyst",
                    "name": "Bob Smith",
                    "background": "Government policy advisor",
                    "expertise": ["regulation", "economics"],
                    "personality": {
                        "risk_appetite": "low",
                        "optimism_bias": "realist",
                        "contrarian_tendency": 0.1,
                        "analytical_depth": "deep",
                    },
                    "initial_stance": "Regulation is necessary for safety",
                    "reasoning_style": "precedent-based",
                },
            ],
        })
        personas = await generate_population(seed, mock_llm, db, count=2)
        assert len(personas) == 2
        assert all(isinstance(p, AgentPersona) for p in personas)

    async def test_generate_population_assigns_ulid_and_archetype(
        self, seed: SeedMaterial, mock_llm, db: Store,
    ):
        """Each persona has ap_ prefix ID and correct archetype."""
        mock_llm.set_response({
            "agents": [
                {"archetype": "contrarian", "name": "Eve"},
            ],
        })
        personas = await generate_population(seed, mock_llm, db, count=1)
        assert len(personas) == 1
        assert personas[0].id.startswith("ap_")
        assert personas[0].archetype == "contrarian"

    async def test_generate_population_persists_to_store(
        self, seed: SeedMaterial, mock_llm, db: Store,
    ):
        """Personas are persisted to the database."""
        mock_llm.set_response({
            "agents": [
                {"archetype": "analyst", "name": "Carol"},
            ],
        })
        personas = await generate_population(seed, mock_llm, db, count=1)
        found = db.get_agent_persona(personas[0].id)
        assert found is not None
        assert found.archetype == "analyst"

    async def test_generate_population_stores_full_persona_json(
        self, seed: SeedMaterial, mock_llm, db: Store,
    ):
        """Full agent dict is stored in persona_json."""
        agent_data = {
            "archetype": "tech_founder",
            "name": "Dave",
            "background": "Founded 3 startups",
            "expertise": ["product", "fundraising"],
        }
        mock_llm.set_response({"agents": [agent_data]})
        personas = await generate_population(seed, mock_llm, db, count=1)
        import json
        stored = json.loads(personas[0].persona_json)
        assert stored["name"] == "Dave"
        assert stored["expertise"] == ["product", "fundraising"]

    async def test_generate_population_uses_prompt_with_seed(
        self, seed: SeedMaterial, mock_llm, db: Store,
    ):
        """The LLM is called with a prompt containing the seed text."""
        mock_llm.set_response({"agents": []})
        await generate_population(seed, mock_llm, db, count=5)
        assert mock_llm.call_count == 1
        last_msg = mock_llm.last_messages[-1]["content"]
        assert "EU announces strict AI agent regulations" in last_msg
        assert "5" in last_msg  # count appears in prompt

    async def test_generate_population_empty_response(
        self, seed: SeedMaterial, mock_llm, db: Store,
    ):
        """Empty agent list from LLM returns empty list."""
        mock_llm.set_response({"agents": []})
        personas = await generate_population(seed, mock_llm, db, count=3)
        assert personas == []

    async def test_generate_population_missing_archetype_defaults(
        self, seed: SeedMaterial, mock_llm, db: Store,
    ):
        """Agent missing archetype gets 'unknown' as default."""
        mock_llm.set_response({
            "agents": [{"name": "No Archetype"}],
        })
        personas = await generate_population(seed, mock_llm, db, count=1)
        assert len(personas) == 1
        assert personas[0].archetype == "unknown"

    async def test_generate_population_requests_sufficient_max_tokens(
        self, seed: SeedMaterial, mock_llm, db: Store,
    ):
        """generate_population requests enough max_tokens for large populations."""
        mock_llm.set_response({"agents": []})
        await generate_population(seed, mock_llm, db, count=30)
        # MockLLMClient stores last call's kwargs — check max_tokens was raised
        assert mock_llm.last_max_tokens >= 4096

    async def test_generate_population_handles_parse_error(
        self, seed: SeedMaterial, db: Store,
    ):
        """When LLM returns truncated JSON (ParseError), return empty list gracefully."""
        from forge.llm.client import MockLLMClient

        mock = MockLLMClient()
        mock.set_parse_error()  # Simulates truncated JSON → ParseError
        personas = await generate_population(seed, mock, db, count=10)
        assert personas == []
