"""Agent population evolution: deactivate underperformers, track leaderboard."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.db.models import AgentPersona
    from forge.db.store import Store


@dataclass
class ReplenishResult:
    """Whether the agent pool needs replenishment."""

    needs_replenishment: bool
    active_count: int
    deficit: int


@dataclass
class AgentEvolutionResult:
    """Summary of an agent evolution cycle."""

    deactivated: list[str] = field(default_factory=list)
    replenish: ReplenishResult | None = None


def deactivate_underperformers(
    store: Store,
    *,
    min_simulations: int = 5,
    min_calibration: float = 0.4,
) -> AgentEvolutionResult:
    """Deactivate personas with poor calibration after sufficient experience.

    Rules (from architecture spec Section 9.2):
    - simulations_participated >= 5 AND calibration_score < 0.4 → deactivated
    """
    result = AgentEvolutionResult()
    personas = store.list_agent_personas(active=True)

    for p in personas:
        if (
            p.simulations_participated >= min_simulations
            and p.calibration_score is not None
            and p.calibration_score < min_calibration
        ):
            store.update_agent_persona(p.id, active=0)
            result.deactivated.append(p.id)

    return result


def get_leaderboard(
    store: Store,
    *,
    limit: int = 10,
) -> list[AgentPersona]:
    """Return top-performing active personas sorted by calibration score."""
    personas = store.list_agent_personas(active=True)
    scored = [p for p in personas if p.calibration_score is not None]
    scored.sort(key=lambda p: p.calibration_score, reverse=True)  # type: ignore[arg-type]
    return scored[:limit]


def replenish_pool(
    store: Store,
    *,
    min_active: int = 10,
) -> ReplenishResult:
    """Check if the active persona pool needs replenishment."""
    active = store.list_agent_personas(active=True)
    active_count = len(active)
    deficit = max(0, min_active - active_count)

    return ReplenishResult(
        needs_replenishment=deficit > 0,
        active_count=active_count,
        deficit=deficit,
    )


def run_agent_evolution(
    store: Store,
    *,
    min_simulations: int = 5,
    min_calibration: float = 0.4,
    min_active: int = 10,
) -> AgentEvolutionResult:
    """Run a full agent evolution cycle: deactivate → check pool."""
    result = deactivate_underperformers(
        store,
        min_simulations=min_simulations,
        min_calibration=min_calibration,
    )
    result.replenish = replenish_pool(store, min_active=min_active)
    return result
