"""Relevance filtering for extracted claims against existing hypothesis graph."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from forge.db.store import Store


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def filter_relevant_claims(
    claims: list[dict],
    store: Store,
    *,
    threshold: float = 0.6,
    duplicate_threshold: float = 0.95,
    _embed_fn: Callable[[str], np.ndarray] | None = None,
) -> list[dict]:
    """Filter claims for relevance, removing near-duplicates.

    A claim passes the filter if:
    - It is novel (no similar hypotheses exist), OR
    - It is related to existing hypotheses (similarity > threshold)

    A claim is rejected if:
    - It is a near-duplicate (similarity > duplicate_threshold)

    Args:
        claims: List of claim dicts with "claim" key.
        store: Database store.
        threshold: Minimum similarity to count as "related".
        duplicate_threshold: Above this similarity, claim is a duplicate.
        _embed_fn: Optional embedding function override (for testing).
    """
    if not claims:
        return []

    # Get all hypotheses with embeddings
    all_hypotheses = store.list_hypotheses()
    embedded = [
        (h, np.frombuffer(h.embedding, dtype=np.float32))
        for h in all_hypotheses
        if h.embedding is not None
    ]

    if not embedded:
        # No existing hypotheses — all claims are novel
        return claims

    # Lazy-load embedder if no test override
    embed_fn = _embed_fn
    if embed_fn is None:
        from forge.retrieve.embeddings import Embedder
        embedder = Embedder()
        embed_fn = embedder.embed

    result = []
    for claim_data in claims:
        claim_text = claim_data.get("claim", "")
        if not claim_text:
            continue

        claim_vec = embed_fn(claim_text)
        max_sim = max(
            _cosine_similarity(claim_vec, vec) for _, vec in embedded
        )

        # Reject near-duplicates
        if max_sim > duplicate_threshold:
            continue

        # Accept: novel or related
        result.append(claim_data)

    return result
