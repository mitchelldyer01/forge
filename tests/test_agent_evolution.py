"""Tests for forge/evolve/agent_evolution.py — Agent population evolution."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from forge.evolve.agent_evolution import (
    AgentEvolutionResult,
    deactivate_underperformers,
    get_leaderboard,
    replenish_pool,
    run_agent_evolution,
)

if TYPE_CHECKING:
    from forge.db.store import Store


# ------------------------------------------------------------------
# Deactivation tests
# ------------------------------------------------------------------


def _make_persona(db: Store, archetype: str, **overrides: object) -> str:
    """Helper to create a persona and update its stats."""
    persona_json = json.dumps({
        "name": f"Test {archetype}",
        "background": "test",
        "expertise": ["test"],
    })
    p = db.save_agent_persona(archetype, persona_json)
    if overrides:
        db.update_agent_persona(p.id, **overrides)
    return p.id


def test_deactivate_low_calibration_experienced(db: Store) -> None:
    """simulations >= 5 AND calibration_score < 0.4 → deactivated."""
    pid = _make_persona(
        db, "bad_predictor",
        simulations_participated=6,
        calibration_score=0.3,
    )

    result = deactivate_underperformers(db)

    persona = db.get_agent_persona(pid)
    assert persona is not None
    assert persona.active == 0
    assert pid in result.deactivated


def test_deactivate_skips_inexperienced(db: Store) -> None:
    """simulations < 5 → not deactivated regardless of calibration."""
    pid = _make_persona(
        db, "new_agent",
        simulations_participated=3,
        calibration_score=0.2,
    )

    deactivate_underperformers(db)

    persona = db.get_agent_persona(pid)
    assert persona is not None
    assert persona.active == 1


def test_deactivate_skips_good_calibration(db: Store) -> None:
    """calibration_score >= 0.4 → not deactivated."""
    pid = _make_persona(
        db, "decent_agent",
        simulations_participated=10,
        calibration_score=0.5,
    )

    deactivate_underperformers(db)

    persona = db.get_agent_persona(pid)
    assert persona is not None
    assert persona.active == 1


def test_deactivate_skips_null_calibration(db: Store) -> None:
    """No calibration score yet → not deactivated."""
    pid = _make_persona(
        db, "unscored",
        simulations_participated=10,
    )

    deactivate_underperformers(db)

    persona = db.get_agent_persona(pid)
    assert persona is not None
    assert persona.active == 1


# ------------------------------------------------------------------
# Leaderboard tests
# ------------------------------------------------------------------


def test_leaderboard_sorted_by_calibration(db: Store) -> None:
    """Leaderboard returns active personas sorted by calibration_score desc."""
    _make_persona(db, "mid", calibration_score=0.6, simulations_participated=5)
    _make_persona(db, "best", calibration_score=0.9, simulations_participated=5)
    _make_persona(db, "worst", calibration_score=0.3, simulations_participated=5)

    leaders = get_leaderboard(db, limit=10)

    assert len(leaders) == 3
    assert leaders[0].archetype == "best"
    assert leaders[1].archetype == "mid"
    assert leaders[2].archetype == "worst"


def test_leaderboard_excludes_inactive(db: Store) -> None:
    """Inactive personas don't appear on leaderboard."""
    _make_persona(db, "active_good", calibration_score=0.8, simulations_participated=5)
    pid = _make_persona(
        db, "inactive_best", calibration_score=0.95, simulations_participated=5,
    )
    db.update_agent_persona(pid, active=0)

    leaders = get_leaderboard(db)

    assert len(leaders) == 1
    assert leaders[0].archetype == "active_good"


def test_leaderboard_limit(db: Store) -> None:
    """Leaderboard respects limit parameter."""
    for i in range(5):
        _make_persona(
            db, f"agent_{i}",
            calibration_score=0.5 + i * 0.1,
            simulations_participated=5,
        )

    leaders = get_leaderboard(db, limit=3)
    assert len(leaders) == 3


def test_leaderboard_excludes_unscored(db: Store) -> None:
    """Personas with no calibration score are excluded."""
    _make_persona(db, "scored", calibration_score=0.7, simulations_participated=5)
    _make_persona(db, "unscored", simulations_participated=5)

    leaders = get_leaderboard(db)
    assert len(leaders) == 1


# ------------------------------------------------------------------
# Replenish pool tests
# ------------------------------------------------------------------


def test_replenish_when_pool_below_threshold(db: Store) -> None:
    """When active pool drops below threshold, flag for replenishment."""
    # Only 2 active personas — below default threshold of 10
    _make_persona(db, "agent_a")
    _make_persona(db, "agent_b")

    result = replenish_pool(db, min_active=10)

    assert result.needs_replenishment is True
    assert result.active_count == 2
    assert result.deficit == 8


def test_replenish_not_needed(db: Store) -> None:
    """Pool above threshold → no replenishment needed."""
    for i in range(12):
        _make_persona(db, f"agent_{i}")

    result = replenish_pool(db, min_active=10)

    assert result.needs_replenishment is False
    assert result.deficit == 0


# ------------------------------------------------------------------
# Full cycle test
# ------------------------------------------------------------------


def test_run_agent_evolution(db: Store) -> None:
    """Full agent evolution cycle."""
    # Good agent
    _make_persona(
        db, "star",
        simulations_participated=10,
        calibration_score=0.85,
    )
    # Bad agent
    bad_id = _make_persona(
        db, "dud",
        simulations_participated=8,
        calibration_score=0.2,
    )

    result = run_agent_evolution(db)

    assert isinstance(result, AgentEvolutionResult)
    assert bad_id in result.deactivated
