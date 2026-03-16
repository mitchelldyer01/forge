"""Tests for claim extraction from articles."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from forge.db.models import Article
    from forge.db.store import Store
    from forge.llm.client import MockLLMClient


# ------------------------------------------------------------------
# Claim extraction
# ------------------------------------------------------------------


@pytest.mark.unit
class TestClaimExtraction:
    def test_extract_claims_from_text_returns_claims(
        self, mock_llm: MockLLMClient,
    ) -> None:
        from forge.extract.claims import extract_claims_from_text

        mock_llm.set_response({
            "claims": [
                {
                    "claim": "AI agents will replace 30% of SaaS tools by 2027",
                    "confidence": 65,
                    "tags": ["ai", "saas"],
                    "resolution_deadline": "2027-01-01",
                },
                {
                    "claim": "Enterprise AI spending will double in 2025",
                    "confidence": 72,
                    "tags": ["ai", "enterprise"],
                    "resolution_deadline": "2025-12-31",
                },
            ]
        })

        import asyncio
        claims = asyncio.run(extract_claims_from_text(
            "AI agents are reshaping enterprise software...", mock_llm,
        ))

        assert len(claims) == 2
        assert claims[0]["claim"] == "AI agents will replace 30% of SaaS tools by 2027"
        assert claims[0]["confidence"] == 65
        assert claims[1]["tags"] == ["ai", "enterprise"]

    def test_extract_claims_from_text_empty_text(
        self, mock_llm: MockLLMClient,
    ) -> None:
        from forge.extract.claims import extract_claims_from_text

        mock_llm.set_response({"claims": []})

        import asyncio
        claims = asyncio.run(extract_claims_from_text("", mock_llm))
        assert claims == []

    def test_extract_claims_from_text_malformed_json(
        self, mock_llm: MockLLMClient,
    ) -> None:
        from forge.extract.claims import extract_claims_from_text

        mock_llm.set_response({"unexpected": "format"})

        import asyncio
        claims = asyncio.run(extract_claims_from_text("some text", mock_llm))
        assert claims == []

    def test_extract_claims_saves_to_db(
        self, db: Store, mock_llm: MockLLMClient, sample_article: Article,
    ) -> None:
        from forge.extract.claims import extract_claims

        mock_llm.set_response({
            "claims": [
                {
                    "claim": "AI agents will replace SaaS tools",
                    "confidence": 65,
                    "tags": ["ai", "saas"],
                    "resolution_deadline": "2027-01-01",
                },
            ]
        })

        import asyncio
        hypotheses = asyncio.run(extract_claims(sample_article, mock_llm, db))

        assert len(hypotheses) == 1
        assert hypotheses[0].claim == "AI agents will replace SaaS tools"
        assert hypotheses[0].source == "rss"
        assert hypotheses[0].source_ref == sample_article.url
        assert hypotheses[0].confidence == 65

    def test_extract_claims_updates_article_count(
        self, db: Store, mock_llm: MockLLMClient, sample_article: Article,
    ) -> None:
        from forge.extract.claims import extract_claims

        mock_llm.set_response({
            "claims": [
                {"claim": "Claim 1", "confidence": 50, "tags": [], "resolution_deadline": None},
                {"claim": "Claim 2", "confidence": 60, "tags": [], "resolution_deadline": None},
            ]
        })

        import asyncio
        asyncio.run(extract_claims(sample_article, mock_llm, db))

        updated = db.get_article(sample_article.id)
        assert updated is not None
        assert updated.claims_extracted == 2

    def test_extract_claims_empty_article_content(
        self, db: Store, mock_llm: MockLLMClient,
    ) -> None:
        from forge.extract.claims import extract_claims

        article = db.save_article(url="https://example.com/empty")
        mock_llm.set_response({"claims": []})

        import asyncio
        hypotheses = asyncio.run(extract_claims(article, mock_llm, db))
        assert hypotheses == []


# ------------------------------------------------------------------
# Prompt loading
# ------------------------------------------------------------------


@pytest.mark.unit
class TestExtractPrompts:
    def test_load_extract_prompt_exists(self) -> None:
        from forge.extract.claims import load_prompt

        prompt = load_prompt(
            "claim_extraction",
            title="Test Title",
            content="Test content about AI agents",
        )
        assert "Test Title" in prompt
        assert "Test content" in prompt

    def test_load_extract_prompt_missing_raises(self) -> None:
        from forge.extract.claims import load_prompt

        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent_prompt")


# ------------------------------------------------------------------
# Relevance filtering
# ------------------------------------------------------------------


@pytest.mark.unit
class TestRelevanceFiltering:
    def test_filter_novel_claims_pass(self, db: Store) -> None:
        """Claims with no similar hypotheses are novel and pass the filter."""
        from forge.extract.relevance import filter_relevant_claims

        claims = [
            {"claim": "Novel claim about quantum computing", "confidence": 60, "tags": []},
        ]
        # No hypotheses in DB — everything is novel
        result = filter_relevant_claims(claims, db, threshold=0.6)
        assert len(result) == 1

    def test_filter_near_duplicates_removed(self, db: Store) -> None:
        """Claims nearly identical to existing hypotheses are filtered out."""
        import numpy as np

        from forge.extract.relevance import filter_relevant_claims
        h = db.save_hypothesis(claim="AI will replace SaaS", source="manual")
        vec = np.random.default_rng(42).random(384).astype(np.float32)
        db.update_hypothesis(h.id, embedding=vec.tobytes())

        # Claim with identical embedding should be filtered as duplicate
        claims = [
            {"claim": "AI will replace SaaS", "confidence": 65, "tags": []},
        ]

        result = filter_relevant_claims(
            claims, db, threshold=0.6, duplicate_threshold=0.95,
            _embed_fn=lambda _text: vec,  # Force identical embedding
        )
        assert len(result) == 0

    def test_filter_related_claims_pass(self, db: Store) -> None:
        """Claims related to existing hypotheses (similarity > threshold) pass."""
        import numpy as np

        from forge.extract.relevance import filter_relevant_claims
        rng = np.random.default_rng(42)
        vec1 = rng.random(384).astype(np.float32)
        vec1 /= np.linalg.norm(vec1)

        h = db.save_hypothesis(claim="Existing AI claim", source="manual")
        db.update_hypothesis(h.id, embedding=vec1.tobytes())

        # Create a vector with ~0.8 similarity (related but not duplicate)
        noise = rng.random(384).astype(np.float32) * 0.3
        vec2 = vec1 + noise
        vec2 /= np.linalg.norm(vec2)

        claims = [{"claim": "Related AI claim", "confidence": 60, "tags": []}]

        result = filter_relevant_claims(
            claims, db, threshold=0.6, duplicate_threshold=0.99,
            _embed_fn=lambda _text: vec2,
        )
        assert len(result) == 1

    def test_filter_empty_claims_returns_empty(self, db: Store) -> None:
        from forge.extract.relevance import filter_relevant_claims

        result = filter_relevant_claims([], db)
        assert result == []

    def test_filter_threshold_configurable(self, db: Store) -> None:
        """Filter uses provided threshold."""
        from forge.extract.relevance import filter_relevant_claims

        claims = [{"claim": "Some claim", "confidence": 50, "tags": []}]
        result = filter_relevant_claims(claims, db, threshold=0.9)
        assert len(result) == 1  # Novel claims still pass
