"""Context injection — retrieve prior hypotheses for analysis enrichment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from forge.retrieve.search import find_similar

if TYPE_CHECKING:
    import numpy as np

    from forge.db.models import Hypothesis
    from forge.db.store import Store


def format_prior_hypotheses(hypotheses: list[Hypothesis]) -> str:
    """Format hypotheses into a string for prompt injection."""
    if not hypotheses:
        return ""
    lines = []
    for h in hypotheses:
        lines.append(
            f"- [{h.id}] \"{h.claim}\" (confidence: {h.confidence}, status: {h.status})"
        )
    return "\n".join(lines)


def format_existing_for_relations(hypotheses: list[Hypothesis]) -> str:
    """Format hypotheses for relation extraction by the judge."""
    if not hypotheses:
        return ""
    lines = []
    for h in hypotheses:
        lines.append(f"- ID: {h.id} | Claim: \"{h.claim}\" | Confidence: {h.confidence}")
    return "\n".join(lines)


def retrieve_prior_context(
    claim_embedding: np.ndarray,
    store: Store,
    *,
    limit: int = 3,
) -> tuple[str, str]:
    """Retrieve prior hypotheses similar to a claim.

    Returns:
        Tuple of (prior_hypotheses_text, existing_hypotheses_text)
        for prompt injection and relation extraction respectively.
    """
    similar = find_similar(claim_embedding, store, limit=limit)
    prior_text = format_prior_hypotheses(similar)
    existing_text = format_existing_for_relations(similar)
    return prior_text, existing_text
