"""Similarity search over hypothesis graph using cosine similarity."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from forge.db.models import Hypothesis
    from forge.db.store import Store


def find_similar(
    query_embedding: np.ndarray,
    store: Store,
    *,
    limit: int = 5,
) -> list[Hypothesis]:
    """Find hypotheses most similar to the query embedding.

    Brute-force cosine similarity over all hypotheses with embeddings.
    Designed for < 10K entries per architecture spec.

    Args:
        query_embedding: 1-D numpy float32 array.
        store: Database store instance.
        limit: Maximum results to return.

    Returns:
        List of Hypothesis models, ordered by descending similarity.
    """
    all_hypotheses = store.list_hypotheses()
    scored: list[tuple[float, Hypothesis]] = []

    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        return []

    for h in all_hypotheses:
        if h.embedding is None:
            continue
        vec = np.frombuffer(h.embedding, dtype=np.float32)
        vec_norm = np.linalg.norm(vec)
        if vec_norm == 0:
            continue
        similarity = float(np.dot(query_embedding, vec) / (query_norm * vec_norm))
        scored.append((similarity, h))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [h for _, h in scored[:limit]]
