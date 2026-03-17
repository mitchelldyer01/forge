"""Agent persona generation from seed material."""

from __future__ import annotations

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


@dataclass
class SeedMaterial:
    """Input for a simulation — the scenario and optional context."""

    text: str
    context: str | None = None


async def generate_population(
    seed: SeedMaterial,
    llm: object,
    store: Store,
    count: int = 30,
) -> list[AgentPersona]:
    """Generate a diverse population of agent personas for simulation.

    Makes one LLM call to generate personas, then persists each to the store.

    Args:
        seed: The scenario seed material.
        llm: LLM client (LLMClient or MockLLMClient).
        store: Database store for persisting personas.
        count: Number of agents to generate.

    Returns:
        List of persisted AgentPersona models.
    """
    prompt = load_swarm_prompt(
        "persona_generator",
        seed_text=seed.text,
        seed_context=seed.context or "",
        count=str(count),
    )

    try:
        response = await llm.complete(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.9,
            max_tokens=max(4096, count * 150),
        )
    except ParseError:
        logger.warning("LLM returned truncated JSON for population generation")
        return []

    agents_data = response.parsed_json or {}
    agent_list = agents_data.get("agents", [])

    personas: list[AgentPersona] = []
    for agent in agent_list:
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
