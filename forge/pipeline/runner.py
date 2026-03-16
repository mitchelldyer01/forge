"""Main pipeline orchestration: ingest → extract → filter."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from forge.calibrate.resolver import check_overdue_predictions
from forge.extract.claims import extract_claims
from forge.ingest.rss import poll_all_feeds
from forge.ingest.url import extract_content

if TYPE_CHECKING:
    from forge.db.store import Store
    from forge.llm.client import LLMClient, MockLLMClient

logger = logging.getLogger(__name__)


async def run_ingestion_cycle(
    store: Store,
    llm: LLMClient | MockLLMClient,
) -> dict[str, int]:
    """Run one ingestion cycle: poll feeds → extract content → extract claims.

    Returns counts: articles_fetched, claims_extracted.
    """
    # Step 1: Poll all due feeds
    feed_results = poll_all_feeds(store)
    total_articles = sum(feed_results.values())

    # Step 2: Fetch content and extract claims from unextracted articles
    total_claims = 0
    for article in store.list_articles(unextracted=True):
        try:
            # Fetch content if missing
            if not article.content and article.url:
                content = extract_content(article.url)
                if content:
                    store.update_article(article.id, content=content)
                    article = store.get_article(article.id)  # type: ignore[assignment]

            # Extract claims
            hypotheses = await extract_claims(article, llm, store)
            total_claims += len(hypotheses)
        except Exception:
            logger.exception("Failed to extract claims from %s", article.url)
            continue

    return {
        "articles_fetched": total_articles,
        "claims_extracted": total_claims,
    }


async def run_pipeline_once(
    store: Store,
    llm: LLMClient | MockLLMClient,
) -> dict:
    """Run a full pipeline cycle: ingest + check overdue predictions."""
    ingestion = await run_ingestion_cycle(store, llm)
    overdue = check_overdue_predictions(store)

    return {
        "ingestion": ingestion,
        "overdue_predictions": len(overdue),
    }
