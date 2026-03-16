"""Tests for forge/retrieve/context.py — Context injection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from forge.retrieve.context import (
    format_existing_for_relations,
    format_prior_hypotheses,
    retrieve_prior_context,
)

if TYPE_CHECKING:
    from forge.db.store import Store


class TestFormatPriorHypotheses:
    @pytest.mark.unit
    def test_empty_list_returns_empty_string(self) -> None:
        assert format_prior_hypotheses([]) == ""

    @pytest.mark.unit
    def test_formats_hypothesis_details(self, db: Store) -> None:
        h = db.save_hypothesis(claim="Test claim", source="manual", confidence=75)
        result = format_prior_hypotheses([h])
        assert h.id in result
        assert "Test claim" in result
        assert "75" in result
        assert "alive" in result


class TestFormatExistingForRelations:
    @pytest.mark.unit
    def test_empty_list_returns_empty_string(self) -> None:
        assert format_existing_for_relations([]) == ""

    @pytest.mark.unit
    def test_includes_id_for_relation_extraction(self, db: Store) -> None:
        h = db.save_hypothesis(claim="Related claim", source="manual")
        result = format_existing_for_relations([h])
        assert h.id in result
        assert "Related claim" in result


class TestRetrievePriorContext:
    @pytest.mark.unit
    def test_retrieves_similar_hypotheses(self, db: Store) -> None:
        # Store hypotheses with embeddings
        h1 = db.save_hypothesis(claim="AI will change everything", source="manual")
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        db.update_hypothesis(h1.id, embedding=vec1.tobytes())

        h2 = db.save_hypothesis(claim="Crypto is volatile", source="manual")
        vec2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        db.update_hypothesis(h2.id, embedding=vec2.tobytes())

        # Query close to AI
        query = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        prior_text, existing_text = retrieve_prior_context(query, db, limit=1)
        assert "AI will change everything" in prior_text
        assert h1.id in existing_text

    @pytest.mark.unit
    def test_empty_db_returns_empty_strings(self, db: Store) -> None:
        query = np.array([1.0, 0.0], dtype=np.float32)
        prior_text, existing_text = retrieve_prior_context(query, db)
        assert prior_text == ""
        assert existing_text == ""

    @pytest.mark.unit
    def test_limit_respected(self, db: Store) -> None:
        for i in range(5):
            h = db.save_hypothesis(claim=f"Claim {i}", source="manual")
            vec = np.random.rand(4).astype(np.float32)
            db.update_hypothesis(h.id, embedding=vec.tobytes())

        query = np.random.rand(4).astype(np.float32)
        prior_text, _ = retrieve_prior_context(query, db, limit=2)
        # Should have exactly 2 lines (2 hypotheses)
        lines = [ln for ln in prior_text.strip().split("\n") if ln.startswith("- ")]
        assert len(lines) == 2
