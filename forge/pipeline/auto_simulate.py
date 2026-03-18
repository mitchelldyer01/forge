"""Auto-simulation: select high-value claims and simulate them."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from forge.swarm.orchestrate import orchestrate_simulation
from forge.swarm.population import SeedMaterial

if TYPE_CHECKING:
    from forge.db.models import Hypothesis
    from forge.db.store import Store

logger = logging.getLogger(__name__)


def select_claims_for_simulation(
    store: Store,
    *,
    top_n: int = 3,
    min_confidence: int = 60,
) -> list[Hypothesis]:
    """Select the best claims for auto-simulation.

    Picks alive hypotheses above min_confidence, sorted by confidence
    descending, capped at top_n.

    Args:
        store: Database store.
        top_n: Maximum claims to return.
        min_confidence: Minimum confidence threshold.

    Returns:
        List of Hypothesis models, sorted by confidence descending.
    """
    candidates = store.list_hypotheses(
        status="alive",
        min_confidence=min_confidence,
    )

    # Sort by confidence descending
    candidates.sort(key=lambda h: h.confidence, reverse=True)

    return candidates[:top_n]


async def auto_simulate_cycle(
    store: Store,
    llm: object,
    *,
    top_n: int = 3,
    min_confidence: int = 60,
    agent_count: int = 14,
    rounds: int = 3,
) -> list[str]:
    """Select top claims and run simulations on them.

    Returns list of simulation IDs created.
    """
    claims = select_claims_for_simulation(
        store, top_n=top_n, min_confidence=min_confidence,
    )

    if not claims:
        logger.info("No claims above threshold for auto-simulation.")
        return []

    sim_ids: list[str] = []
    for claim in claims:
        seed = SeedMaterial(text=claim.claim, context=claim.context)
        try:
            result = await orchestrate_simulation(
                seed, llm, store,
                agent_count=agent_count, rounds=rounds,
            )
            sim_ids.append(result.simulation.id)
            logger.info(
                "Auto-simulated claim: '%s' → simulation %s",
                claim.claim[:80], result.simulation.id,
            )
        except Exception:
            logger.exception("Auto-simulation failed for claim: %s", claim.claim[:80])
            continue

    return sim_ids
