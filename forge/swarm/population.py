"""Agent persona generation from seed material."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from forge.llm.client import ParseError
from forge.swarm.prompts import load_swarm_prompt

if TYPE_CHECKING:
    from forge.db.models import AgentPersona
    from forge.db.store import Store

logger = logging.getLogger(__name__)

BATCH_SIZE = 5


@dataclass
class SeedMaterial:
    """Input for a simulation — the scenario and optional context."""

    text: str
    context: str | None = None


async def _generate_batch(
    seed: SeedMaterial,
    llm: object,
    batch_count: int,
) -> list[dict]:
    """Generate a single batch of agent personas via one LLM call.

    Returns a list of raw agent dicts, or empty list on failure.
    """
    prompt = load_swarm_prompt(
        "persona_generator",
        seed_text=seed.text,
        seed_context=seed.context or "",
        count=str(batch_count),
    )

    try:
        response = await llm.complete(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.9,
            max_tokens=max(2048, batch_count * 300),
        )
    except ParseError as e:
        logger.warning(
            "LLM returned invalid JSON for batch of %d agents: %s",
            batch_count,
            e,
        )
        return []

    agents_data = response.parsed_json or {}
    return agents_data.get("agents", [])


async def generate_population(
    seed: SeedMaterial,
    llm: object,
    store: Store,
    count: int = 30,
) -> list[AgentPersona]:
    """Generate a diverse population of agent personas for simulation.

    Splits the requested count into batches of BATCH_SIZE and runs them
    concurrently. Each batch is an independent LLM call, which avoids
    truncation on large populations and gives the LLM more focus per agent.

    Args:
        seed: The scenario seed material.
        llm: LLM client (LLMClient or MockLLMClient).
        store: Database store for persisting personas.
        count: Number of agents to generate.

    Returns:
        List of persisted AgentPersona models.
    """
    # Split into batches
    batch_sizes = []
    remaining = count
    while remaining > 0:
        batch = min(BATCH_SIZE, remaining)
        batch_sizes.append(batch)
        remaining -= batch

    # Run all batches concurrently
    tasks = [
        _generate_batch(seed, llm, batch_count)
        for batch_count in batch_sizes
    ]
    batch_results = await asyncio.gather(*tasks)

    # Flatten and persist
    all_agents: list[dict] = []
    for agents in batch_results:
        all_agents.extend(agents)

    personas: list[AgentPersona] = []
    for agent in all_agents:
        archetype = agent.get("archetype", "unknown")
        try:
            persona = store.save_agent_persona(
                archetype=archetype,
                persona_json=json.dumps(agent),
            )
            personas.append(persona)
        except Exception:
            logger.warning("Failed to save agent persona: %s", archetype)

    return personas
