"""Claim extraction from articles via LLM."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Template

if TYPE_CHECKING:
    from forge.db.models import Article, Hypothesis
    from forge.db.store import Store
    from forge.llm.client import LLMClient, MockLLMClient

_PROMPTS_DIR = Path(__file__).parent / "prompts"
logger = logging.getLogger(__name__)


def load_prompt(name: str, **variables: str) -> str:
    """Load and render an extract prompt template by name."""
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    template = Template(path.read_text())
    return template.render(**variables)


async def extract_claims_from_text(
    text: str,
    llm: LLMClient | MockLLMClient,
) -> list[dict]:
    """Extract raw claims from arbitrary text via LLM.

    Returns list of dicts with keys: claim, confidence, tags, resolution_deadline.
    """
    if not text:
        prompt = load_prompt("claim_extraction", title="", content="")
    else:
        prompt = load_prompt("claim_extraction", title="", content=text)

    response = await llm.complete(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    parsed = response.parsed_json
    if not parsed or "claims" not in parsed:
        logger.warning("LLM returned unexpected format for claim extraction")
        return []

    return parsed["claims"]


async def extract_claims(
    article: Article,
    llm: LLMClient | MockLLMClient,
    store: Store,
) -> list[Hypothesis]:
    """Extract claims from an article and save as hypotheses."""
    content = article.content or ""
    title = article.title or ""

    raw_claims = await extract_claims_from_text(
        f"{title}\n\n{content}" if content else title, llm,
    )

    hypotheses = []
    for claim_data in raw_claims:
        claim_text = claim_data.get("claim", "")
        if not claim_text:
            continue

        h = store.save_hypothesis(
            claim=claim_text,
            source="rss",
            source_ref=article.url,
            confidence=claim_data.get("confidence", 50),
            tags=claim_data.get("tags"),
            resolution_deadline=claim_data.get("resolution_deadline"),
        )
        hypotheses.append(h)

    store.update_article(article.id, claims_extracted=len(hypotheses))
    return hypotheses
