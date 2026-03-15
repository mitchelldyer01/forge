"""Persistence wiring — save pipeline output to DB."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.analyze.structured import Verdict
    from forge.db.models import Hypothesis
    from forge.db.store import Store


def save_verdict(
    store: Store,
    *,
    claim: str,
    verdict: Verdict,
    context: str | None = None,
    source: str = "manual",
) -> Hypothesis:
    """Save a structured analysis verdict as a hypothesis in the DB."""
    return store.save_hypothesis(
        claim=claim,
        source=source,
        context=context,
        confidence=verdict.confidence,
        tags=verdict.tags,
    )
