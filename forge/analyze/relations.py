"""Save relations extracted by the judge from verdict output."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.analyze.structured import Verdict
    from forge.db.store import Store

logger = logging.getLogger(__name__)


def save_verdict_relations(
    source_hypothesis_id: str,
    verdict: Verdict,
    store: Store,
) -> None:
    """Persist relations from a verdict to the store.

    Args:
        source_hypothesis_id: The hypothesis that was just analyzed.
        verdict: The analysis verdict, which may contain relations.
        store: Database store instance.
    """
    if not verdict.relations:
        return

    for rel in verdict.relations:
        target_id = rel.get("target_id", "")
        relation_type = rel.get("type", "")
        reasoning = rel.get("reasoning", "")

        if not target_id or not relation_type:
            logger.warning("Skipping relation with missing target_id or type: %s", rel)
            continue

        store.save_relation(
            source_id=source_hypothesis_id,
            target_id=target_id,
            relation_type=relation_type,
            reasoning=reasoning,
        )
        logger.info(
            "Saved relation: %s -[%s]-> %s",
            source_hypothesis_id, relation_type, target_id,
        )
