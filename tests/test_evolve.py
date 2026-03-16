"""Tests for forge/evolve/selection.py — Hypothesis evolution (cull/promote/fork)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from forge.evolve.selection import (
    EvolutionResult,
    cull_hypotheses,
    fork_hypotheses,
    promote_hypotheses,
    run_evolution_cycle,
)

if TYPE_CHECKING:
    from forge.db.store import Store


# ------------------------------------------------------------------
# Culling tests
# ------------------------------------------------------------------


def test_cull_low_confidence_old_unchallenged(db: Store) -> None:
    """confidence < 25, challenges_survived < 2, age > 7 days → dead."""
    h = db.save_hypothesis("weak claim", "manual", confidence=20)
    # Backdate created_at to 10 days ago
    old_date = (datetime.now(UTC) - timedelta(days=10)).isoformat()
    db.update_hypothesis(h.id, created_at=old_date)

    result = cull_hypotheses(db)

    updated = db.get_hypothesis(h.id)
    assert updated is not None
    assert updated.status == "dead"
    assert h.id in result.culled


def test_cull_skips_young_hypothesis(db: Store) -> None:
    """confidence < 25 but age < 7 days → stays alive."""
    h = db.save_hypothesis("new weak claim", "manual", confidence=20)

    result = cull_hypotheses(db)

    updated = db.get_hypothesis(h.id)
    assert updated is not None
    assert updated.status == "alive"
    assert h.id not in result.culled


def test_cull_skips_challenged_hypothesis(db: Store) -> None:
    """confidence < 25, old, but challenges_survived >= 2 → stays alive."""
    h = db.save_hypothesis("challenged claim", "manual", confidence=20)
    old_date = (datetime.now(UTC) - timedelta(days=10)).isoformat()
    db.update_hypothesis(h.id, created_at=old_date, challenges_survived=3)

    result = cull_hypotheses(db)

    updated = db.get_hypothesis(h.id)
    assert updated is not None
    assert updated.status == "alive"
    assert h.id not in result.culled


def test_cull_human_rejected(db: Store) -> None:
    """human_rejected >= 3 AND human_endorsed == 0 → dead."""
    h = db.save_hypothesis("rejected claim", "manual", confidence=60)
    db.update_hypothesis(h.id, human_rejected=3, human_endorsed=0)

    result = cull_hypotheses(db)

    updated = db.get_hypothesis(h.id)
    assert updated is not None
    assert updated.status == "dead"
    assert h.id in result.culled


def test_cull_human_rejected_with_endorsements_stays(db: Store) -> None:
    """human_rejected >= 3 but human_endorsed > 0 → stays alive."""
    h = db.save_hypothesis("controversial claim", "manual", confidence=60)
    db.update_hypothesis(h.id, human_rejected=3, human_endorsed=1)

    cull_hypotheses(db)

    updated = db.get_hypothesis(h.id)
    assert updated is not None
    assert updated.status == "alive"


def test_cull_dormant_no_activity(db: Store) -> None:
    """No activity in 30 days → dormant."""
    h = db.save_hypothesis("stale claim", "manual", confidence=50)
    old_date = (datetime.now(UTC) - timedelta(days=35)).isoformat()
    # Use raw SQL to backdate since update_hypothesis always sets updated_at=now
    db.conn.execute(
        "UPDATE hypotheses SET updated_at = ?, created_at = ? WHERE id = ?",
        (old_date, old_date, h.id),
    )
    db.conn.commit()

    result = cull_hypotheses(db)

    updated = db.get_hypothesis(h.id)
    assert updated is not None
    assert updated.status == "dormant"
    assert h.id in result.dormant


def test_cull_resolved_false_becomes_dead(db: Store) -> None:
    """resolved_false → dead."""
    h = db.save_hypothesis("wrong claim", "manual", status="resolved_false")

    result = cull_hypotheses(db)

    updated = db.get_hypothesis(h.id)
    assert updated is not None
    assert updated.status == "dead"
    assert h.id in result.culled


def test_cull_skips_already_dead(db: Store) -> None:
    """Already dead hypotheses are not processed."""
    h = db.save_hypothesis("dead claim", "manual", status="dead", confidence=10)

    result = cull_hypotheses(db)

    assert h.id not in result.culled
    assert h.id not in result.dormant


# ------------------------------------------------------------------
# Promotion tests
# ------------------------------------------------------------------


def test_promote_high_conviction(db: Store) -> None:
    """confidence >= 75 AND challenges_survived >= 3 → tag 'high_conviction'."""
    h = db.save_hypothesis("strong claim", "manual", confidence=80, tags=["ai"])
    db.update_hypothesis(h.id, challenges_survived=4)

    result = promote_hypotheses(db)

    updated = db.get_hypothesis(h.id)
    assert updated is not None
    assert "high_conviction" in (updated.tags or [])
    assert h.id in result.promoted


def test_promote_high_conviction_already_tagged(db: Store) -> None:
    """Already tagged with high_conviction → no duplicate tag."""
    h = db.save_hypothesis(
        "strong claim", "manual", confidence=80, tags=["high_conviction"]
    )
    db.update_hypothesis(h.id, challenges_survived=5)

    promote_hypotheses(db)

    updated = db.get_hypothesis(h.id)
    assert updated is not None
    assert updated.tags.count("high_conviction") == 1


def test_promote_human_endorsed_boosts_confidence(db: Store) -> None:
    """human_endorsed >= 2 → confidence += 10, capped at 95."""
    h = db.save_hypothesis("endorsed claim", "manual", confidence=70)
    db.update_hypothesis(h.id, human_endorsed=2)

    result = promote_hypotheses(db)

    updated = db.get_hypothesis(h.id)
    assert updated is not None
    assert updated.confidence == 80
    assert h.id in result.boosted


def test_promote_human_endorsed_caps_at_95(db: Store) -> None:
    """Confidence boost caps at 95."""
    h = db.save_hypothesis("very confident claim", "manual", confidence=90)
    db.update_hypothesis(h.id, human_endorsed=3)

    promote_hypotheses(db)

    updated = db.get_hypothesis(h.id)
    assert updated is not None
    assert updated.confidence == 95


def test_promote_resolved_true_tags_confirmed(db: Store) -> None:
    """resolved_true → tag 'confirmed'."""
    h = db.save_hypothesis("correct claim", "manual", status="resolved_true")

    result = promote_hypotheses(db)

    updated = db.get_hypothesis(h.id)
    assert updated is not None
    assert "confirmed" in (updated.tags or [])
    assert h.id in result.confirmed


# ------------------------------------------------------------------
# Forking tests
# ------------------------------------------------------------------


def test_fork_top_hypotheses(db: Store) -> None:
    """Top 10% by confidence get forked."""
    # Create 10 hypotheses with varying confidence
    for i in range(10):
        db.save_hypothesis(f"claim {i}", "manual", confidence=30 + i * 7)

    result = fork_hypotheses(db)

    # Top 10% of 10 = 1 hypothesis should be forked
    assert len(result.forked) >= 1
    # Forked hypotheses should exist in DB
    for fork_id in result.forked:
        forked = db.get_hypothesis(fork_id)
        assert forked is not None
        assert forked.source == "fork"
        assert forked.parent_id is not None
        assert forked.generation > 0


def test_fork_preserves_parent_lineage(db: Store) -> None:
    """Forked hypothesis has correct parent_id and incremented generation."""
    h = db.save_hypothesis("top claim", "manual", confidence=95)

    result = fork_hypotheses(db)

    for fork_id in result.forked:
        forked = db.get_hypothesis(fork_id)
        assert forked is not None
        assert forked.parent_id == h.id
        assert forked.generation == 1


def test_fork_empty_db_returns_empty(db: Store) -> None:
    """No hypotheses → no forks."""
    result = fork_hypotheses(db)
    assert len(result.forked) == 0


# ------------------------------------------------------------------
# Full cycle test
# ------------------------------------------------------------------


def test_run_evolution_cycle(db: Store) -> None:
    """Full evolution cycle: cull + promote + fork in order."""
    # Cull target: old low-confidence
    h1 = db.save_hypothesis("weak old", "manual", confidence=15)
    old_date = (datetime.now(UTC) - timedelta(days=10)).isoformat()
    db.update_hypothesis(h1.id, created_at=old_date)

    # Promote target: high confidence + challenged
    h2 = db.save_hypothesis("strong", "manual", confidence=85)
    db.update_hypothesis(h2.id, challenges_survived=5)

    # Promote target: endorsed
    h3 = db.save_hypothesis("endorsed", "manual", confidence=60)
    db.update_hypothesis(h3.id, human_endorsed=3)

    result = run_evolution_cycle(db)

    assert isinstance(result, EvolutionResult)
    assert h1.id in result.culled
    assert h2.id in result.promoted
    assert h3.id in result.boosted


def test_evolution_cycle_skips_non_alive(db: Store) -> None:
    """Dead/dormant hypotheses aren't culled again or promoted."""
    h = db.save_hypothesis("already dead", "manual", status="dead", confidence=10)

    result = run_evolution_cycle(db)

    assert h.id not in result.culled
    assert h.id not in result.promoted
