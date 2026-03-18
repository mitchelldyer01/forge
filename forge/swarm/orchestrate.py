"""Headless simulation orchestration — no UI dependencies."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from forge.retrieve.context import format_prior_hypotheses
from forge.retrieve.search import find_similar
from forge.swarm.arena import SimulationResult, run_simulation
from forge.swarm.population import SeedMaterial, generate_population
from forge.swarm.predictions import extract_predictions

if TYPE_CHECKING:
    from collections.abc import Callable

    from forge.db.models import Prediction
    from forge.db.store import Store

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationResult:
    """Result of a headless simulation orchestration."""

    simulation: object
    predictions: list[Prediction] = field(default_factory=list)


def inject_prior_knowledge(
    seed: SeedMaterial,
    store: Store,
    *,
    limit: int = 5,
    _embed_fn: Callable[[str], np.ndarray] | None = None,
) -> SeedMaterial:
    """Enrich seed with prior hypotheses from the knowledge base.

    Embeds the seed text, finds similar hypotheses, and appends them
    to the seed context so agents are aware of prior analysis.

    Args:
        seed: The original seed material.
        store: Database store.
        limit: Max prior hypotheses to inject.
        _embed_fn: Override embedding function (for testing).

    Returns:
        New SeedMaterial with enriched context, or original if no
        prior knowledge found.
    """
    embed_fn = _embed_fn
    if embed_fn is None:
        try:
            from forge.retrieve.embeddings import get_embedder
            embed_fn = get_embedder().embed
        except Exception:
            logger.debug("Embeddings not available, skipping prior knowledge injection.")
            return seed

    query_embedding = embed_fn(seed.text)
    if np.linalg.norm(query_embedding) == 0:
        return seed

    similar = find_similar(query_embedding, store, limit=limit)
    if not similar:
        return seed

    prior_text = format_prior_hypotheses(similar)
    if not prior_text:
        return seed

    existing_context = seed.context or ""
    separator = "\n\n" if existing_context else ""
    enriched_context = (
        f"{existing_context}{separator}"
        f"## Prior Analysis\n"
        f"The following related hypotheses have been previously analyzed:\n"
        f"{prior_text}"
    )

    logger.info(
        "Injected %d prior hypotheses into simulation context.",
        len(similar),
    )

    return SeedMaterial(text=seed.text, context=enriched_context)


async def orchestrate_simulation(
    seed: SeedMaterial,
    llm: object,
    store: Store,
    *,
    agent_count: int = 14,
    rounds: int = 3,
    _embed_fn: Callable[[str], np.ndarray] | None = None,
) -> OrchestrationResult:
    """Run a complete simulation: population → debate → consensus → predictions.

    Injects prior knowledge from the hypothesis graph before generating
    the agent population. This is the headless equivalent of the CLI's
    _run_simulate(), without Rich progress bars or console output.

    Args:
        seed: The scenario seed material.
        llm: LLM client (LLMClient or MockLLMClient).
        store: Database store.
        agent_count: Number of agents to generate.
        rounds: Number of debate rounds.
        _embed_fn: Override embedding function (for testing).

    Returns:
        OrchestrationResult with simulation and predictions.

    Raises:
        RuntimeError: If no agents could be generated.
    """
    # Enrich seed with prior knowledge
    seed = inject_prior_knowledge(seed, store, _embed_fn=_embed_fn)

    population = await generate_population(seed, llm, store, count=agent_count)
    if not population:
        raise RuntimeError(
            f"No agents generated — LLM failed to produce valid personas "
            f"(requested {agent_count}). Check LLM server output."
        )

    sim_result: SimulationResult = await run_simulation(
        seed, population, llm, store,
        rounds=rounds, max_concurrent=2,
    )

    predictions = await extract_predictions(
        seed, sim_result.consensus, llm, store, sim_result.simulation.id,
        agent_count=sim_result.simulation.agent_count,
    )

    logger.info(
        "Simulation %s complete: %d agents, %d rounds, %d predictions",
        sim_result.simulation.id, len(population), rounds, len(predictions),
    )

    return OrchestrationResult(
        simulation=sim_result.simulation,
        predictions=predictions,
    )
