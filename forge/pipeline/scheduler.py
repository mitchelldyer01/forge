"""Cron-like scheduling for pipeline cycles."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from forge.pipeline.runner import run_pipeline_once

if TYPE_CHECKING:
    from forge.db.store import Store
    from forge.llm.client import LLMClient, MockLLMClient

logger = logging.getLogger(__name__)


async def run_scheduled_once(
    store: Store,
    llm: LLMClient | MockLLMClient,
) -> dict:
    """Run a single scheduled pipeline cycle. Returns summary."""
    return await run_pipeline_once(store, llm)


async def run_scheduled(
    store: Store,
    llm: LLMClient | MockLLMClient,
    interval_minutes: int = 240,
) -> None:
    """Run pipeline cycles on a recurring interval. Runs indefinitely."""
    while True:
        try:
            result = await run_pipeline_once(store, llm)
            ingestion = result["ingestion"]
            logger.info(
                "Pipeline cycle complete: %d articles, %d claims, %d overdue",
                ingestion["articles_fetched"],
                ingestion["claims_extracted"],
                result["overdue_predictions"],
            )
        except Exception:
            logger.exception("Pipeline cycle failed")

        await asyncio.sleep(interval_minutes * 60)
