"""Prediction extraction from simulation consensus output."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from forge.swarm.prompts import load_swarm_prompt

if TYPE_CHECKING:
    from forge.db.models import Prediction
    from forge.db.store import Store
    from forge.swarm.consensus import ConsensusReport
    from forge.swarm.population import SeedMaterial

logger = logging.getLogger(__name__)


def format_consensus_for_prompt(consensus: ConsensusReport) -> str:
    """Format a ConsensusReport as readable text for LLM prompt injection."""
    lines = [
        f"Majority position: {consensus.majority_position}",
        f"Majority confidence: {consensus.majority_confidence:.0f}%",
        f"Majority fraction: {consensus.majority_fraction:.0%} of agents",
    ]

    if consensus.dissent_clusters:
        lines.append("\nDissent clusters:")
        for cluster in consensus.dissent_clusters:
            lines.append(
                f"- {cluster.position}: {cluster.agent_count} agents, "
                f"avg confidence {cluster.avg_confidence:.0f}%"
            )
            for arg in cluster.key_arguments:
                lines.append(f"  - {arg}")

    if consensus.conviction_shifts:
        lines.append("\nConviction shifts:")
        for shift in consensus.conviction_shifts:
            lines.append(
                f"- {shift.archetype}: {shift.from_position} → {shift.to_position} "
                f"(delta: {shift.confidence_delta:+d})"
            )

    if consensus.edge_cases:
        lines.append("\nEdge cases:")
        for ec in consensus.edge_cases:
            lines.append(f"- {ec.archetype} ({ec.position}): {ec.reasoning}")

    return "\n".join(lines)


async def extract_predictions(
    seed: SeedMaterial,
    consensus: ConsensusReport,
    llm: object,
    store: Store,
    simulation_id: str,
    agent_count: int | None = None,
) -> list[Prediction]:
    """Extract predictions from simulation consensus via one LLM call.

    Args:
        seed: The original scenario seed material.
        consensus: The consensus report from the simulation.
        llm: LLM client (LLMClient or MockLLMClient).
        store: Database store for persistence.
        simulation_id: The simulation ID to associate predictions with.
        agent_count: Number of agents in simulation (scales token budget).

    Returns:
        List of persisted Prediction models.
    """
    consensus_text = format_consensus_for_prompt(consensus)
    prompt = load_swarm_prompt(
        "extraction",
        seed_text=seed.text,
        consensus_report=consensus_text,
    )

    # Scale token budget for larger simulations
    max_tokens = max(2048, (agent_count or 0) * 100)

    response = await llm.complete(
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.5,
        max_tokens=max_tokens,
    )

    data = response.parsed_json or {}
    prediction_list = data.get("predictions", [])

    predictions: list[Prediction] = []
    for pred in prediction_list:
        claim = pred.get("claim", "")
        if not claim:
            logger.warning("Skipping prediction with missing claim")
            continue

        confidence = pred.get("confidence", 50)
        try:
            prediction = store.save_prediction(
                simulation_id=simulation_id,
                claim=claim,
                confidence=confidence,
                consensus_strength=pred.get("consensus_strength"),
                dissent_summary=pred.get("dissent_summary"),
                resolution_deadline=pred.get("resolution_deadline"),
            )
            predictions.append(prediction)
        except Exception:
            logger.warning("Failed to save prediction: %s", claim)

    # Update simulation predictions count
    if predictions:
        store.update_simulation(
            simulation_id, predictions_extracted=len(predictions),
        )

    return predictions
