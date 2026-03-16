"""Daily digest generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from forge.evolve.agent_evolution import get_leaderboard

if TYPE_CHECKING:
    from forge.db.models import AgentPersona, Hypothesis, Prediction, Relation
    from forge.db.store import Store


@dataclass
class Digest:
    """Daily digest content."""

    generated_at: str
    new_predictions: list[Prediction] = field(default_factory=list)
    resolved_predictions: list[Prediction] = field(default_factory=list)
    high_conviction: list[Hypothesis] = field(default_factory=list)
    killed: list[Hypothesis] = field(default_factory=list)
    contradictions: list[tuple[Relation, str, str]] = field(default_factory=list)
    agent_leaderboard: list[AgentPersona] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render digest as markdown."""
        lines = [
            "# FORGE Daily Digest",
            f"*Generated: {self.generated_at}*",
            "",
        ]

        # New predictions
        lines.append("## New Predictions")
        if self.new_predictions:
            for p in self.new_predictions:
                deadline = p.resolution_deadline or "no deadline"
                lines.append(
                    f"- **{p.claim}** "
                    f"(confidence: {p.confidence}, deadline: {deadline})"
                )
        else:
            lines.append("*No new predictions.*")
        lines.append("")

        # Resolved predictions
        lines.append("## Resolved Predictions")
        if self.resolved_predictions:
            for p in self.resolved_predictions:
                lines.append(
                    f"- **{p.claim}** → {p.resolved_as} "
                    f"(was {p.confidence}% confident)"
                )
        else:
            lines.append("*No resolved predictions.*")
        lines.append("")

        # High conviction
        lines.append("## High-Conviction Hypotheses")
        if self.high_conviction:
            for h in self.high_conviction:
                lines.append(f"- **{h.claim}** ({h.confidence}%)")
        else:
            lines.append("*None.*")
        lines.append("")

        # Killed
        lines.append("## Killed Hypotheses")
        if self.killed:
            for h in self.killed:
                lines.append(f"- ~~{h.claim}~~")
        else:
            lines.append("*None.*")
        lines.append("")

        # Contradictions
        lines.append("## Active Contradictions")
        if self.contradictions:
            for _rel, claim_a, claim_b in self.contradictions:
                lines.append(f"- **{claim_a}** ↔ **{claim_b}**")
        else:
            lines.append("*None.*")
        lines.append("")

        # Agent leaderboard
        lines.append("## Agent Leaderboard")
        if self.agent_leaderboard:
            for i, a in enumerate(self.agent_leaderboard, 1):
                score = (
                    f"{a.calibration_score:.0%}"
                    if a.calibration_score is not None
                    else "N/A"
                )
                lines.append(
                    f"{i}. **{a.archetype}** — {score} accuracy "
                    f"({a.simulations_participated} sims)"
                )
        else:
            lines.append("*No scored agents yet.*")

        return "\n".join(lines) + "\n"


def generate_digest(store: Store) -> Digest:
    """Generate a daily digest from current DB state."""
    now = datetime.now(UTC).isoformat()

    # New predictions (all pending)
    new_predictions = store.list_predictions_pending()

    # Resolved predictions
    resolved_predictions = store.list_resolved_predictions()

    # High conviction hypotheses
    all_alive = store.list_hypotheses(status="alive")
    high_conviction = [
        h for h in all_alive
        if h.tags and "high_conviction" in h.tags
    ]

    # Killed hypotheses
    killed = store.list_hypotheses(status="dead")

    # Active contradictions
    contradictions: list[tuple] = []
    for h in all_alive:
        relations = store.list_relations_for_hypothesis(h.id)
        for r in relations:
            if r.relation_type == "contradicts" and r.source_id == h.id:
                target = store.get_hypothesis(r.target_id)
                if target and target.status == "alive":
                    contradictions.append(
                        (r, h.claim, target.claim)
                    )

    # Agent leaderboard (top 5)
    agent_leaderboard = get_leaderboard(store, limit=5)

    return Digest(
        generated_at=now,
        new_predictions=new_predictions,
        resolved_predictions=resolved_predictions,
        high_conviction=high_conviction,
        killed=killed,
        contradictions=contradictions,
        agent_leaderboard=agent_leaderboard,
    )
