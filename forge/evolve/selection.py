"""Hypothesis evolution: cull, promote, and fork."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.db.store import Store


@dataclass
class EvolutionResult:
    """Summary of a single evolution cycle."""

    culled: list[str] = field(default_factory=list)
    dormant: list[str] = field(default_factory=list)
    promoted: list[str] = field(default_factory=list)
    boosted: list[str] = field(default_factory=list)
    confirmed: list[str] = field(default_factory=list)
    forked: list[str] = field(default_factory=list)


def cull_hypotheses(
    store: Store,
    *,
    min_confidence: int = 25,
    min_age_days: int = 7,
    dormant_days: int = 30,
) -> EvolutionResult:
    """Apply culling rules to alive hypotheses.

    Rules (from architecture spec Section 9.1):
    - confidence < 25 AND challenges_survived < 2 AND age > 7 days → dead
    - human_rejected >= 3 AND human_endorsed == 0 → dead
    - No activity in 30 days → dormant
    - resolved_false → dead
    """
    result = EvolutionResult()
    now = datetime.now(UTC)
    age_cutoff = (now - timedelta(days=min_age_days)).isoformat()
    dormant_cutoff = (now - timedelta(days=dormant_days)).isoformat()

    hypotheses = store.list_hypotheses(status="alive")

    # Also process resolved_false (they may not be status="alive")
    resolved_false = [
        h for h in store.list_hypotheses()
        if h.status == "resolved_false"
    ]

    for h in resolved_false:
        store.update_hypothesis(h.id, status="dead")
        result.culled.append(h.id)

    for h in hypotheses:
        # Rule 1: low confidence + old + unchallenged
        if (
            h.confidence < min_confidence
            and h.challenges_survived < 2
            and h.created_at < age_cutoff
        ):
            store.update_hypothesis(h.id, status="dead")
            result.culled.append(h.id)
            continue

        # Rule 2: human rejected without endorsements
        if h.human_rejected >= 3 and h.human_endorsed == 0:
            store.update_hypothesis(h.id, status="dead")
            result.culled.append(h.id)
            continue

        # Rule 3: dormant (no activity in 30 days)
        if h.updated_at < dormant_cutoff:
            store.update_hypothesis(h.id, status="dormant")
            result.dormant.append(h.id)
            continue

    return result


def promote_hypotheses(store: Store) -> EvolutionResult:
    """Apply promotion rules to alive hypotheses.

    Rules (from architecture spec Section 9.1):
    - confidence >= 75 AND challenges_survived >= 3 → tag 'high_conviction'
    - human_endorsed >= 2 → confidence += 10 (cap 95)
    - resolved_true → tag 'confirmed'
    """
    result = EvolutionResult()

    # Process alive hypotheses for high_conviction and endorsement boost
    hypotheses = store.list_hypotheses(status="alive")

    for h in hypotheses:
        tags = list(h.tags or [])

        # Rule 1: high conviction tagging
        if h.confidence >= 75 and h.challenges_survived >= 3:
            if "high_conviction" not in tags:
                tags.append("high_conviction")
            store.update_hypothesis(h.id, tags=json.dumps(tags))
            result.promoted.append(h.id)

        # Rule 2: endorsement boost
        if h.human_endorsed >= 2:
            new_confidence = min(h.confidence + 10, 95)
            store.update_hypothesis(h.id, confidence=new_confidence)
            result.boosted.append(h.id)

    # Process resolved_true hypotheses for 'confirmed' tag
    all_hypotheses = store.list_hypotheses()
    for h in all_hypotheses:
        if h.status == "resolved_true":
            tags = list(h.tags or [])
            if "confirmed" not in tags:
                tags.append("confirmed")
                store.update_hypothesis(h.id, tags=json.dumps(tags))
            result.confirmed.append(h.id)

    return result


def fork_hypotheses(store: Store) -> EvolutionResult:
    """Fork the top 10% of alive hypotheses by confidence.

    Creates variant hypotheses with:
    - source="fork", parent_id set, generation incremented
    - Slightly modified claims (stronger/adjacent/temporal variants)
    """
    result = EvolutionResult()

    hypotheses = store.list_hypotheses(status="alive")
    if not hypotheses:
        return result

    # Sort by confidence descending
    sorted_h = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)

    # Top 10% (at least 1)
    top_count = max(1, len(sorted_h) // 10)
    top_hypotheses = sorted_h[:top_count]

    for h in top_hypotheses:
        # Create a "stronger" variant
        variant_claim = f"[Stronger variant] {h.claim}"
        tags = list(h.tags or [])

        forked = store.save_hypothesis(
            claim=variant_claim,
            source="fork",
            context=h.context,
            confidence=h.confidence,
            tags=tags if tags else None,
            parent_id=h.id,
            source_ref=h.id,
        )
        # Update generation
        store.update_hypothesis(forked.id, generation=h.generation + 1)
        result.forked.append(forked.id)

    return result


def run_evolution_cycle(store: Store) -> EvolutionResult:
    """Run a full evolution cycle: cull → promote → fork."""
    cull_result = cull_hypotheses(store)
    promote_result = promote_hypotheses(store)
    fork_result = fork_hypotheses(store)

    return EvolutionResult(
        culled=cull_result.culled,
        dormant=cull_result.dormant,
        promoted=promote_result.promoted,
        boosted=promote_result.boosted,
        confirmed=promote_result.confirmed,
        forked=fork_result.forked,
    )
