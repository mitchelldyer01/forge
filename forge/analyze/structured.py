"""Structured analysis pipeline — steelman → redteam → judge."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel

from forge.analyze.prompts import load_prompt

if TYPE_CHECKING:
    from forge.llm.client import CompletionResponse, LLMClient, MockLLMClient

logger = logging.getLogger(__name__)


class AnalysisError(Exception):
    """Raised when the analysis pipeline fails."""

    def __init__(self, stage: str, original_error: Exception) -> None:
        self.stage = stage
        self.original_error = original_error
        super().__init__(f"Analysis failed at {stage}: {original_error}")


class Verdict(BaseModel):
    """Result of structured analysis."""

    position: str
    confidence: int
    synthesis: str
    steelman_arg: str
    redteam_arg: str
    conditions: list[str]
    tags: list[str]
    relations: list[dict] | None = None


async def analyze(
    claim: str,
    llm: LLMClient | MockLLMClient,
    *,
    context: str | None = None,
    prior_hypotheses: str | None = None,
    existing_hypotheses: str | None = None,
) -> Verdict:
    """Run three-pass structured analysis on a claim.

    Args:
        claim: The hypothesis to analyze.
        llm: LLM client (real or mock).
        context: Optional background context.
        prior_hypotheses: Formatted string of similar prior hypotheses.
        existing_hypotheses: Formatted string for relation extraction.

    Returns:
        Verdict with position, confidence, and supporting analysis.

    Raises:
        AnalysisError: If any stage of the pipeline fails.
    """
    # Stage 1: Steelman
    try:
        steelman_prompt = load_prompt(
            "steelman", claim=claim, context=context or "",
            prior_hypotheses=prior_hypotheses or "",
        )
        steelman_resp = await _call_llm(llm, steelman_prompt)
        steelman_output = json.dumps(steelman_resp.parsed_json, indent=2)
    except Exception as e:
        raise AnalysisError("steelman", e) from e

    # Stage 2: Redteam
    try:
        redteam_prompt = load_prompt(
            "redteam", claim=claim, context=context or "",
            steelman_output=steelman_output,
            prior_hypotheses=prior_hypotheses or "",
        )
        redteam_resp = await _call_llm(llm, redteam_prompt)
        redteam_output = json.dumps(redteam_resp.parsed_json, indent=2)
    except Exception as e:
        raise AnalysisError("redteam", e) from e

    # Stage 3: Judge
    try:
        judge_prompt = load_prompt(
            "judge", claim=claim, context=context or "",
            steelman_output=steelman_output,
            redteam_output=redteam_output,
            prior_hypotheses=prior_hypotheses or "",
            existing_hypotheses=existing_hypotheses or "",
        )
        judge_resp = await _call_llm(llm, judge_prompt)
        judge_data = judge_resp.parsed_json or {}
    except Exception as e:
        raise AnalysisError("judge", e) from e

    return Verdict(
        position=judge_data.get("position", ""),
        confidence=judge_data.get("confidence", 50),
        synthesis=judge_data.get("synthesis", ""),
        steelman_arg=judge_data.get("steelman_arg", ""),
        redteam_arg=judge_data.get("redteam_arg", ""),
        conditions=judge_data.get("conditions", []),
        tags=judge_data.get("tags", []),
        relations=judge_data.get("relations"),
    )


async def _call_llm(
    llm: LLMClient | MockLLMClient,
    system_prompt: str,
) -> CompletionResponse:
    """Make a single LLM call with the given system prompt."""
    return await llm.complete(
        messages=[
            {"role": "system", "content": system_prompt},
        ],
        response_format={"type": "json_object"},
    )
