"""Tests for pipeline runner and scheduling."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from forge.db.store import Store
    from forge.llm.client import MockLLMClient


# ------------------------------------------------------------------
# Pipeline runner
# ------------------------------------------------------------------


@pytest.mark.integration
class TestPipelineRunner:
    def test_run_ingestion_cycle_with_feeds(
        self, db: Store, mock_llm: MockLLMClient,
    ) -> None:
        import asyncio

        from forge.pipeline.runner import run_ingestion_cycle

        # Set up a feed
        db.save_feed(name="Test", url="https://test.com/feed", poll_interval_minutes=0)

        # Mock feedparser to return articles
        mock_parsed = MagicMock()
        mock_entries = [
            MagicMock(
                link="https://test.com/post1",
                title="Post 1",
                get=lambda k, d=None, _t="Post 1": _t if k == "title" else d,
                published_parsed=None,
            ),
        ]
        mock_parsed.entries = mock_entries
        mock_parsed.bozo = False

        # Mock LLM for claim extraction
        mock_llm.set_response({"claims": [
            {
                "claim": "Test claim", "confidence": 60,
                "tags": ["test"], "resolution_deadline": None,
            },
        ]})

        # Mock trafilatura for content extraction
        with (
            patch("forge.ingest.rss.feedparser.parse", return_value=mock_parsed),
            patch("forge.ingest.url.trafilatura.fetch_url", return_value="<p>content</p>"),
            patch("forge.ingest.url.trafilatura.extract", return_value="Article content"),
        ):
            result = asyncio.run(run_ingestion_cycle(db, mock_llm))

        assert result["articles_fetched"] >= 1
        assert "claims_extracted" in result

    def test_run_ingestion_cycle_no_feeds(
        self, db: Store, mock_llm: MockLLMClient,
    ) -> None:
        import asyncio

        from forge.pipeline.runner import run_ingestion_cycle

        result = asyncio.run(run_ingestion_cycle(db, mock_llm))
        assert result["articles_fetched"] == 0
        assert result["claims_extracted"] == 0

    def test_run_pipeline_once_returns_summary(
        self, db: Store, mock_llm: MockLLMClient,
    ) -> None:
        import asyncio

        from forge.pipeline.runner import run_pipeline_once

        result = asyncio.run(run_pipeline_once(db, mock_llm))
        assert "ingestion" in result
        assert "overdue_predictions" in result

    def test_run_pipeline_once_reports_overdue(
        self, db: Store, mock_llm: MockLLMClient,
    ) -> None:
        import asyncio
        from datetime import UTC, datetime, timedelta

        from forge.pipeline.runner import run_pipeline_once

        sim = db.save_simulation(mode="scenario", seed_text="test")
        past = (datetime.now(UTC) - timedelta(days=30)).isoformat()
        db.save_prediction(
            simulation_id=sim.id,
            claim="Overdue prediction",
            confidence=70,
            resolution_deadline=past,
        )

        result = asyncio.run(run_pipeline_once(db, mock_llm))
        assert result["overdue_predictions"] >= 1

    def test_run_ingestion_cycle_error_in_extraction_continues(
        self, db: Store, mock_llm: MockLLMClient,
    ) -> None:
        """If claim extraction fails for one article, pipeline continues."""
        import asyncio

        from forge.pipeline.runner import run_ingestion_cycle

        db.save_feed(name="Test", url="https://test.com/feed", poll_interval_minutes=0)

        mock_parsed = MagicMock()
        mock_entries = [
            MagicMock(
                link="https://test.com/post1",
                title="Post 1",
                get=lambda k, d=None, _t="Post 1": _t if k == "title" else d,
                published_parsed=None,
            ),
        ]
        mock_parsed.entries = mock_entries
        mock_parsed.bozo = False

        # LLM returns error
        mock_llm.set_error(500)

        with (
            patch("forge.ingest.rss.feedparser.parse", return_value=mock_parsed),
            patch("forge.ingest.url.trafilatura.fetch_url", return_value="<p>content</p>"),
            patch("forge.ingest.url.trafilatura.extract", return_value="Content"),
        ):
            result = asyncio.run(run_ingestion_cycle(db, mock_llm))

        # Should not crash — articles fetched but extraction failed
        assert result["articles_fetched"] >= 1


# ------------------------------------------------------------------
# Scheduler
# ------------------------------------------------------------------


@pytest.mark.unit
class TestScheduler:
    def test_scheduler_runs_cycle(
        self, db: Store, mock_llm: MockLLMClient,
    ) -> None:
        import asyncio

        from forge.pipeline.scheduler import run_scheduled_once

        result = asyncio.run(run_scheduled_once(db, mock_llm))
        assert isinstance(result, dict)
