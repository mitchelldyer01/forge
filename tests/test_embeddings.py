"""Tests for forge/retrieve/embeddings.py — Embedding generation and storage."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

if TYPE_CHECKING:
    from forge.db.store import Store


class TestEmbedder:
    @pytest.mark.unit
    def test_embed_returns_numpy_array(self) -> None:
        with patch("forge.retrieve.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.rand(384).astype(np.float32)
            mock_st.return_value = mock_model

            from forge.retrieve.embeddings import Embedder

            embedder = Embedder()
            result = embedder.embed("test text")
            assert isinstance(result, np.ndarray)
            assert result.ndim == 1
            assert result.dtype == np.float32

    @pytest.mark.unit
    def test_embed_calls_model_encode(self) -> None:
        with patch("forge.retrieve.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.zeros(384, dtype=np.float32)
            mock_st.return_value = mock_model

            from forge.retrieve.embeddings import Embedder

            embedder = Embedder()
            embedder.embed("hello world")
            mock_model.encode.assert_called_once_with(
                "hello world", convert_to_numpy=True
            )

    @pytest.mark.unit
    def test_embed_consistent_dimensions(self) -> None:
        with patch("forge.retrieve.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.rand(384).astype(np.float32)
            mock_st.return_value = mock_model

            from forge.retrieve.embeddings import Embedder

            embedder = Embedder()
            r1 = embedder.embed("first")
            r2 = embedder.embed("second")
            assert r1.shape == r2.shape

    @pytest.mark.unit
    def test_get_embedder_caches_instance(self) -> None:
        """get_embedder() returns the same Embedder instance on repeated calls."""
        with patch("forge.retrieve.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model

            from forge.retrieve.embeddings import _clear_cache, get_embedder

            _clear_cache()
            e1 = get_embedder()
            e2 = get_embedder()
            assert e1 is e2
            # SentenceTransformer should only be constructed once
            assert mock_st.call_count == 1
            _clear_cache()

    @pytest.mark.unit
    def test_get_embedder_different_model_creates_new(self) -> None:
        """get_embedder() with a different model name creates a new instance."""
        with patch("forge.retrieve.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model

            from forge.retrieve.embeddings import _clear_cache, get_embedder

            _clear_cache()
            e1 = get_embedder()
            e2 = get_embedder("other-model")
            assert e1 is not e2
            assert mock_st.call_count == 2
            _clear_cache()


class TestEmbeddingStorage:
    @pytest.mark.unit
    def test_store_hypothesis_embedding_roundtrip(self, db: Store) -> None:
        h = db.save_hypothesis(claim="Test embedding", source="manual")
        vec = np.random.rand(384).astype(np.float32)
        db.update_hypothesis(h.id, embedding=vec.tobytes())
        updated = db.get_hypothesis(h.id)
        assert updated is not None
        assert updated.embedding is not None
        recovered = np.frombuffer(updated.embedding, dtype=np.float32)
        assert np.allclose(vec, recovered)

    @pytest.mark.unit
    def test_embed_and_store_multiple(self, db: Store) -> None:
        claims = ["AI claim", "Crypto claim", "Climate claim"]
        for claim in claims:
            h = db.save_hypothesis(claim=claim, source="manual")
            vec = np.random.rand(384).astype(np.float32)
            db.update_hypothesis(h.id, embedding=vec.tobytes())

        hypotheses = db.list_hypotheses()
        embedded_count = sum(1 for h in hypotheses if h.embedding is not None)
        assert embedded_count == 3

    @pytest.mark.unit
    def test_hypothesis_without_embedding(self, db: Store) -> None:
        h = db.save_hypothesis(claim="No embedding", source="manual")
        loaded = db.get_hypothesis(h.id)
        assert loaded is not None
        assert loaded.embedding is None
