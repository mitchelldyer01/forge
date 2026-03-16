"""Tests for forge/retrieve/search.py — Similarity search over hypothesis graph."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from forge.retrieve.search import find_similar

if TYPE_CHECKING:
    from forge.db.store import Store


def _store_with_embeddings(db: Store) -> list[str]:
    """Create hypotheses with embeddings and return their IDs."""
    # Create distinct vectors for different topics
    ai_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    crypto_vec = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    climate_vec = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    mixed_vec = np.array([0.7, 0.3, 0.0, 0.0], dtype=np.float32)  # AI-ish

    ids = []
    for claim, vec in [
        ("AI will transform software", ai_vec),
        ("Bitcoin will reach 100k", crypto_vec),
        ("Climate change accelerating", climate_vec),
        ("AI agents in crypto trading", mixed_vec),
    ]:
        h = db.save_hypothesis(claim=claim, source="manual")
        db.update_hypothesis(h.id, embedding=vec.tobytes())
        ids.append(h.id)
    return ids


class TestFindSimilar:
    @pytest.mark.unit
    def test_find_similar_returns_ordered_results(self, db: Store) -> None:
        _store_with_embeddings(db)
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = find_similar(query, db, limit=4)
        # First result should be "AI will transform software" (exact match)
        assert results[0].claim == "AI will transform software"
        # Second should be the mixed AI/crypto one
        assert results[1].claim == "AI agents in crypto trading"

    @pytest.mark.unit
    def test_find_similar_respects_limit(self, db: Store) -> None:
        _store_with_embeddings(db)
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = find_similar(query, db, limit=2)
        assert len(results) == 2

    @pytest.mark.unit
    def test_find_similar_empty_db(self, db: Store) -> None:
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = find_similar(query, db, limit=5)
        assert results == []

    @pytest.mark.unit
    def test_find_similar_skips_unembedded(self, db: Store) -> None:
        db.save_hypothesis(claim="No embedding", source="manual")
        h2 = db.save_hypothesis(claim="Has embedding", source="manual")
        vec = np.array([1.0, 0.0], dtype=np.float32)
        db.update_hypothesis(h2.id, embedding=vec.tobytes())

        query = np.array([1.0, 0.0], dtype=np.float32)
        results = find_similar(query, db, limit=5)
        assert len(results) == 1
        assert results[0].claim == "Has embedding"
