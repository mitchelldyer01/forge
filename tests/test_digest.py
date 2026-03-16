"""Tests for forge/export/digest.py — Daily digest generation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from forge.export.digest import generate_digest

if TYPE_CHECKING:
    from forge.db.store import Store


def test_digest_empty_db(db: Store) -> None:
    """Empty database produces a valid digest with zero counts."""
    digest = generate_digest(db)

    assert digest.new_predictions == []
    assert digest.resolved_predictions == []
    assert digest.high_conviction == []
    assert digest.killed == []
    assert digest.contradictions == []
    assert digest.agent_leaderboard == []


def test_digest_new_predictions(db: Store) -> None:
    """Predictions created in last 24h appear in new_predictions."""
    sim = db.save_simulation("scenario", "test")
    p = db.save_prediction(sim.id, "AI pricing shifts", 72)

    digest = generate_digest(db)

    assert len(digest.new_predictions) == 1
    assert digest.new_predictions[0].id == p.id


def test_digest_resolved_predictions(db: Store) -> None:
    """Resolved predictions appear in resolved_predictions."""
    sim = db.save_simulation("scenario", "test")
    p = db.save_prediction(sim.id, "BTC rises", 65)
    now = datetime.now(UTC).isoformat()
    db.update_prediction(p.id, resolved_as="true", resolved_at=now)

    digest = generate_digest(db)

    assert len(digest.resolved_predictions) == 1
    assert digest.resolved_predictions[0].resolved_as == "true"


def test_digest_high_conviction(db: Store) -> None:
    """Hypotheses tagged high_conviction appear in digest."""
    db.save_hypothesis(
        "Strong claim", "manual",
        confidence=85, tags=["high_conviction"],
    )

    digest = generate_digest(db)

    assert len(digest.high_conviction) == 1


def test_digest_killed_hypotheses(db: Store) -> None:
    """Dead hypotheses appear in killed list."""
    db.save_hypothesis("Dead claim", "manual", status="dead")

    digest = generate_digest(db)

    assert len(digest.killed) == 1


def test_digest_contradictions(db: Store) -> None:
    """Pairs with 'contradicts' relation appear."""
    h1 = db.save_hypothesis("Claim A", "manual", confidence=60)
    h2 = db.save_hypothesis("Claim B", "manual", confidence=55)
    db.save_relation(h1.id, h2.id, "contradicts")

    digest = generate_digest(db)

    assert len(digest.contradictions) >= 1


def test_digest_agent_leaderboard(db: Store) -> None:
    """Top agents by calibration appear in leaderboard."""
    persona_json = json.dumps({"name": "Star Agent", "background": "test"})
    p = db.save_agent_persona("star_analyst", persona_json)
    db.update_agent_persona(
        p.id, calibration_score=0.85, simulations_participated=10,
    )

    digest = generate_digest(db)

    assert len(digest.agent_leaderboard) == 1
    assert digest.agent_leaderboard[0].archetype == "star_analyst"


def test_digest_render_markdown(db: Store) -> None:
    """Digest renders to readable markdown."""
    db.save_hypothesis("Test claim", "manual", confidence=50)
    sim = db.save_simulation("scenario", "test")
    db.save_prediction(sim.id, "Test prediction", 60)

    digest = generate_digest(db)
    md = digest.to_markdown()

    assert "# FORGE Daily Digest" in md
    assert "Test prediction" in md
