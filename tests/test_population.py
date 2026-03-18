"""
Tests for forge.swarm.population — agent persona generation from seed material.

Mirrors: forge/swarm/population.py
"""

import pytest

from forge.db.models import AgentPersona
from forge.db.store import Store
from forge.swarm.population import (
    BATCH_SIZE,
    SeedMaterial,
    _deduplicate_agents,
    generate_population,
)


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
        await generate_population(seed, mock_llm, db, count=BATCH_SIZE)
        assert mock_llm.call_count >= 1
        last_msg = mock_llm.last_messages[-1]["content"]
        assert "EU announces strict AI agent regulations" in last_msg

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
        """Each batch requests enough max_tokens for its agents."""
        mock_llm.set_response({"agents": []})
        await generate_population(seed, mock_llm, db, count=BATCH_SIZE)
        assert mock_llm.last_max_tokens >= 2048

    async def test_generate_population_requests_600_tokens_per_agent(
        self, seed: SeedMaterial, mock_llm, db: Store,
    ):
        """Batch of 5 agents requests at least 3000 max_tokens (600 per agent)."""
        mock_llm.set_response({"agents": []})
        await generate_population(seed, mock_llm, db, count=BATCH_SIZE)
        assert mock_llm.last_max_tokens >= 3000

    async def test_generate_population_handles_parse_error(
        self, seed: SeedMaterial, db: Store,
    ):
        """When LLM returns truncated JSON (ParseError) twice, return empty list."""
        from forge.llm.client import MockLLMClient

        mock = MockLLMClient()
        # Queue two parse errors — first attempt + retry
        mock._parse_errors = [True, True]
        personas = await generate_population(seed, mock, db, count=BATCH_SIZE)
        assert personas == []


@pytest.mark.unit
class TestDeduplicateAgents:
    """Tests for _deduplicate_agents — post-generation archetype dedup."""

    def test_deduplicate_removes_same_archetype(self):
        """Two agents with same archetype are deduplicated to one."""
        agents = [
            {"archetype": "retail_investor", "name": "Alice",
             "initial_stance": "Regulation helps investors"},
            {"archetype": "retail_investor", "name": "Bob",
             "initial_stance": "Regulation protects consumers"},
        ]
        result = _deduplicate_agents(agents)
        assert len(result) == 1
        assert result[0]["name"] == "Alice"

    def test_deduplicate_keeps_different_archetypes(self):
        """Agents with different archetypes are all kept."""
        agents = [
            {"archetype": "retail_investor", "name": "Alice"},
            {"archetype": "policy_analyst", "name": "Bob"},
            {"archetype": "tech_optimist", "name": "Carol"},
        ]
        result = _deduplicate_agents(agents)
        assert len(result) == 3

    def test_deduplicate_normalizes_case_and_whitespace(self):
        """Archetype comparison is case-insensitive and whitespace-normalized."""
        agents = [
            {"archetype": "Retail_Investor", "name": "Alice"},
            {"archetype": "retail_investor", "name": "Bob"},
            {"archetype": "RETAIL_INVESTOR", "name": "Carol"},
        ]
        result = _deduplicate_agents(agents)
        assert len(result) == 1

    def test_deduplicate_empty_list(self):
        """Empty input returns empty output."""
        assert _deduplicate_agents([]) == []

    def test_deduplicate_missing_archetype(self):
        """Agents without archetype key are kept (each treated as unique)."""
        agents = [
            {"name": "Alice"},
            {"archetype": "analyst", "name": "Bob"},
        ]
        result = _deduplicate_agents(agents)
        assert len(result) == 2


@pytest.mark.unit
class TestGeneratePopulationBatching:
    """Tests for batched population generation."""

    @pytest.fixture
    def seed(self) -> SeedMaterial:
        return SeedMaterial(
            text="The EU announces strict AI agent regulations",
            context="European Commission proposal",
        )

    async def test_batches_large_count_into_multiple_calls(
        self, seed: SeedMaterial, mock_llm, db: Store,
    ):
        """count > BATCH_SIZE results in multiple LLM calls."""
        count = BATCH_SIZE * 2
        batch_responses = [
            {"agents": [
                {"archetype": f"type_{i}", "name": f"Agent {i}"}
                for i in range(j, min(j + BATCH_SIZE, count))
            ]}
            for j in range(0, count, BATCH_SIZE)
        ]
        mock_llm.set_responses(batch_responses)
        personas = await generate_population(seed, mock_llm, db, count=count)
        assert mock_llm.call_count == 2
        assert len(personas) == count

    async def test_uneven_batch_handles_remainder(
        self, seed: SeedMaterial, mock_llm, db: Store,
    ):
        """count not evenly divisible by BATCH_SIZE still generates all."""
        count = BATCH_SIZE + 2
        batch_responses = [
            {"agents": [
                {"archetype": f"type_{i}", "name": f"Agent {i}"}
                for i in range(BATCH_SIZE)
            ]},
            {"agents": [
                {"archetype": f"type_{i}", "name": f"Agent {i}"}
                for i in range(BATCH_SIZE, count)
            ]},
        ]
        mock_llm.set_responses(batch_responses)
        personas = await generate_population(seed, mock_llm, db, count=count)
        assert mock_llm.call_count == 2
        assert len(personas) == count

    async def test_single_batch_for_small_count(
        self, seed: SeedMaterial, mock_llm, db: Store,
    ):
        """count <= BATCH_SIZE uses a single LLM call."""
        mock_llm.set_response({
            "agents": [{"archetype": "a", "name": "X"}],
        })
        await generate_population(seed, mock_llm, db, count=1)
        assert mock_llm.call_count == 1

    async def test_parse_error_in_one_batch_preserves_others(
        self, seed: SeedMaterial, db: Store,
    ):
        """A failed batch doesn't lose agents from successful batches."""
        import json

        from forge.llm.client import (
            CompletionResponse,
            MockLLMClient,
            ParseError,
        )

        mock = MockLLMClient()
        count = BATCH_SIZE + 1  # forces 2 batches
        call_counter = 0

        async def mixed_complete(messages, **kwargs):
            nonlocal call_counter
            call_counter += 1
            if call_counter == 1:
                data = {"agents": [
                    {"archetype": "analyst", "name": "Alice"},
                ]}
                return CompletionResponse(
                    content=json.dumps(data),
                    parsed_json=data,
                    token_count=50,
                    raw_response={"mock": True},
                )
            raise ParseError("truncated", ValueError("bad"))

        mock.complete = mixed_complete
        personas = await generate_population(seed, mock, db, count=count)
        # Should still have the agents from the first batch
        assert len(personas) >= 1
        assert personas[0].archetype == "analyst"

    async def test_generate_population_deduplicates_across_batches(
        self, seed: SeedMaterial, mock_llm, db: Store,
    ):
        """Duplicate archetypes across batches are removed before persisting."""
        count = BATCH_SIZE * 2
        batch_responses = [
            {"agents": [
                {"archetype": "retail_investor", "name": "Alice",
                 "initial_stance": "Regulation helps investors"},
                {"archetype": "policy_analyst", "name": "Bob",
                 "initial_stance": "Regulation is needed"},
            ]},
            {"agents": [
                {"archetype": "retail_investor", "name": "Carol",
                 "initial_stance": "Regulation protects consumers"},
                {"archetype": "tech_optimist", "name": "Dave",
                 "initial_stance": "Innovation will thrive"},
            ]},
        ]
        mock_llm.set_responses(batch_responses)
        personas = await generate_population(seed, mock_llm, db, count=count)
        archetypes = [p.archetype for p in personas]
        # retail_investor appears in both batches with similar stance — deduped
        assert archetypes.count("retail_investor") == 1
        # unique archetypes kept
        assert "policy_analyst" in archetypes
        assert "tech_optimist" in archetypes

    async def test_generate_population_caps_at_requested_count(
        self, seed: SeedMaterial, mock_llm, db: Store,
    ):
        """Population is capped at requested count even if LLM over-generates."""
        # Request 3, but mock returns 5 per batch (2 batches = 10 total)
        count = 3
        batch_responses = [
            {"agents": [
                {"archetype": f"type_a{i}", "name": f"Agent A{i}"}
                for i in range(5)
            ]},
            {"agents": [
                {"archetype": f"type_b{i}", "name": f"Agent B{i}"}
                for i in range(5)
            ]},
        ]
        mock_llm.set_responses(batch_responses)
        personas = await generate_population(seed, mock_llm, db, count=count)
        assert len(personas) <= count

    async def test_generate_population_backfills_on_shortfall(
        self, seed: SeedMaterial, mock_llm, db: Store,
    ):
        """When dedup/failure leaves fewer agents than requested, backfill."""
        # Request 5, but first batch only returns 3 (simulating under-delivery)
        # Backfill call returns the remaining 2
        mock_llm.set_responses([
            {"agents": [
                {"archetype": "analyst", "name": "A"},
                {"archetype": "optimist", "name": "B"},
                {"archetype": "skeptic", "name": "C"},
            ]},
            {"agents": [
                {"archetype": "regulator", "name": "D"},
                {"archetype": "founder", "name": "E"},
            ]},
        ])
        personas = await generate_population(seed, mock_llm, db, count=5)
        assert len(personas) == 5

    async def test_generate_population_backfill_limited_to_two_attempts(
        self, seed: SeedMaterial, mock_llm, db: Store,
    ):
        """Backfill gives up after 2 attempts — no infinite loop."""
        # Request 5, first batch returns 3, backfill always returns empty
        mock_llm.set_responses([
            {"agents": [
                {"archetype": "analyst", "name": "A"},
                {"archetype": "optimist", "name": "B"},
                {"archetype": "skeptic", "name": "C"},
            ]},
            {"agents": []},
            {"agents": []},
        ])
        personas = await generate_population(seed, mock_llm, db, count=5)
        # Should return 3 (what it has), not loop forever
        assert len(personas) == 3
        assert mock_llm.call_count == 3  # 1 initial + 2 backfill attempts

    async def test_generate_batch_passes_exclusion_to_prompt(
        self, seed: SeedMaterial, mock_llm, db: Store,
    ):
        """Excluded archetypes appear in the LLM prompt."""
        from forge.swarm.population import _generate_batch

        mock_llm.set_response({"agents": []})
        await _generate_batch(
            seed, mock_llm, 3, exclude=["retail_investor", "policy_analyst"],
        )
        sent = mock_llm.last_messages[-1]["content"]
        assert "retail_investor" in sent
        assert "policy_analyst" in sent

    async def test_generate_population_sequential_with_exclusion(
        self, seed: SeedMaterial, mock_llm, db: Store,
    ):
        """Second batch receives first batch's archetypes as exclusion."""
        count = BATCH_SIZE * 2
        batch_responses = [
            {"agents": [
                {"archetype": f"type_{i}", "name": f"Agent {i}"}
                for i in range(BATCH_SIZE)
            ]},
            {"agents": [
                {"archetype": f"type_{i + BATCH_SIZE}", "name": f"Agent {i + BATCH_SIZE}"}
                for i in range(BATCH_SIZE)
            ]},
        ]
        mock_llm.set_responses(batch_responses)
        await generate_population(seed, mock_llm, db, count=count)

        # Second call's prompt should contain first batch's archetypes
        assert mock_llm.call_count == 2
        second_prompt = mock_llm.all_messages[1][-1]["content"]
        assert "type_0" in second_prompt

    async def test_generate_population_retries_failed_batch(
        self, seed: SeedMaterial, db: Store,
    ):
        """A failed batch is retried once before giving up."""
        import json

        from forge.llm.client import (
            CompletionResponse,
            MockLLMClient,
            ParseError,
        )

        mock = MockLLMClient()
        call_counter = 0

        async def retry_complete(messages, **kwargs):
            nonlocal call_counter
            call_counter += 1
            if call_counter == 1:
                # First call fails
                raise ParseError("truncated", ValueError("bad"))
            # Retry succeeds
            data = {"agents": [
                {"archetype": "analyst", "name": "Bob"},
            ]}
            return CompletionResponse(
                content=json.dumps(data),
                parsed_json=data,
                token_count=50,
                raw_response={"mock": True},
            )

        mock.complete = retry_complete
        personas = await generate_population(seed, mock, db, count=1)
        assert call_counter == 2  # First call failed, second succeeded
        assert len(personas) == 1
        assert personas[0].archetype == "analyst"
