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
