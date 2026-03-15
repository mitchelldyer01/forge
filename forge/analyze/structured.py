"""Structured analysis pipeline: steelman -> redteam -> judge."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import BaseModel

from forge.analyze.prompts import load_prompt

if TYPE_CHECKING:
    from forge.llm.client import LLMClient


class Verdict(BaseModel):
    """Result of a structured analysis pipeline run."""

    position: str  # support | oppose | conditional
    confidence: int  # 0-100
    synthesis: str
    steelman_arg: str
    redteam_arg: str
    conditions: list[str] | None = None
    tags: list[str] | None = None


async def run_structured_analysis(
    llm: LLMClient,
    *,
    claim: str,
    context: str | None = None,
) -> Verdict:
    """Run steelman -> redteam -> judge pipeline on a claim.

    Args:
        llm: LLM client (real or mock).
        claim: The claim to analyze.
        context: Optional background context.

    Returns:
        A Verdict with position, confidence, synthesis, and both arguments.
    """
    # Step 1: Steelman
    steelman_prompt = load_prompt("steelman", claim=claim, context=context)
    steelman_resp = await llm.complete(
        [{"role": "user", "content": steelman_prompt}],
        response_format={"type": "json_object"},
    )
    steelman_arg = json.dumps(steelman_resp.content)

    # Step 2: Redteam (receives steelman output)
    redteam_prompt = load_prompt(
        "redteam", claim=claim, context=context, steelman_arg=steelman_arg,
    )
    redteam_resp = await llm.complete(
        [{"role": "user", "content": redteam_prompt}],
        response_format={"type": "json_object"},
    )
    redteam_arg = json.dumps(redteam_resp.content)

    # Step 3: Judge (receives both)
    judge_prompt = load_prompt(
        "judge", claim=claim, context=context,
        steelman_arg=steelman_arg, redteam_arg=redteam_arg,
    )
    judge_resp = await llm.complete(
        [{"role": "user", "content": judge_prompt}],
        response_format={"type": "json_object"},
    )
    judge = judge_resp.content

    return Verdict(
        position=judge.get("position", "conditional"),
        confidence=judge.get("confidence", 50),
        synthesis=judge.get("synthesis", ""),
        steelman_arg=steelman_arg,
        redteam_arg=redteam_arg,
        conditions=judge.get("conditions") or None,
        tags=judge.get("tags") or None,
    )
