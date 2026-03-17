"""Simulation output rendering — round summaries and result panels."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PositionStats:
    """Aggregated stats for a single position in a round."""

    count: int
    avg_confidence: float


@dataclass
class RoundSummary:
    """Summary of a single simulation round."""

    round_num: int
    responded: int
    expected_agents: int
    failed: int
    changed: int
    positions: dict[str, PositionStats] = field(default_factory=dict)

    def format_line(self) -> str:
        """Format as compact terminal output."""
        header = f"Round {self.round_num} — {self.responded}/{self.expected_agents} agents"
        extras = []
        if self.changed > 0:
            extras.append(f"{self.changed} changed")
        if self.failed > 0:
            extras.append(f"{self.failed} failed")
        if extras:
            header += f"  [{', '.join(extras)}]"

        # Position breakdown, sorted by count descending
        parts = []
        for pos, stats in sorted(
            self.positions.items(), key=lambda x: x[1].count, reverse=True,
        ):
            parts.append(f"{pos}: {stats.count} (avg {stats.avg_confidence:.0f}%)")

        lines = [header]
        if parts:
            lines.append("  " + "  ".join(parts))
        return "\n".join(lines)


def summarize_round(
    turns: list,
    *,
    round_num: int,
    expected_agents: int,
    prev_turns: list | None = None,
) -> RoundSummary:
    """Compute a RoundSummary from a list of turns.

    Args:
        turns: Successful turns for this round (SimulationTurn-like objects
            with .position, .confidence, .agent_persona_id).
        round_num: Which round (1, 2, 3).
        expected_agents: Total agents in the simulation.
        prev_turns: Turns from the previous round, used to count position changes.
    """
    responded = len(turns)
    failed = expected_agents - responded

    # Group by position
    pos_groups: dict[str, list[int]] = {}
    for turn in turns:
        pos = turn.position or "neutral"
        pos_groups.setdefault(pos, []).append(turn.confidence or 0)

    positions = {
        pos: PositionStats(
            count=len(confs),
            avg_confidence=sum(confs) / len(confs) if confs else 0.0,
        )
        for pos, confs in pos_groups.items()
    }

    # Count changed positions
    changed = 0
    if prev_turns:
        prev_by_agent = {t.agent_persona_id: t.position for t in prev_turns}
        for turn in turns:
            prev_pos = prev_by_agent.get(turn.agent_persona_id)
            if prev_pos is not None and prev_pos != turn.position:
                changed += 1

    return RoundSummary(
        round_num=round_num,
        responded=responded,
        expected_agents=expected_agents,
        failed=failed,
        changed=changed,
        positions=positions,
    )
