"""Debate digest — surface novel arguments for Round 3 convergence."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from forge.swarm.interaction import compute_novelty_score

if TYPE_CHECKING:
    from forge.db.models import AgentPersona, SimulationTurn


def build_debate_digest(
    r2_turns: list[SimulationTurn],
    exclude_agent_id: str,
    personas_map: dict[str, AgentPersona],
    max_items: int = 5,
) -> str:
    """Build a digest of the most novel R2 arguments for convergence context.

    Excludes the current agent's own turn and ranks remaining turns by
    novelty score to surface unique perspectives.

    Args:
        r2_turns: All Round 2 turns from the simulation.
        exclude_agent_id: The current agent's persona ID (excluded).
        personas_map: Dict mapping persona ID to AgentPersona.
        max_items: Maximum number of digest entries.

    Returns:
        Formatted string with one line per argument, or empty string.
    """
    others = [t for t in r2_turns if t.agent_persona_id != exclude_agent_id]
    if not others:
        return ""

    scored = [
        (compute_novelty_score(t, r2_turns), t)
        for t in others
    ]

    # Position-balanced selection: guarantee at least one entry per position
    by_position: dict[str, list[tuple[float, SimulationTurn]]] = {}
    for score, turn in scored:
        pos = turn.position or "neutral"
        by_position.setdefault(pos, []).append((score, turn))
    for pos in by_position:
        by_position[pos].sort(key=lambda x: x[0], reverse=True)

    # Phase 1: top-novelty entry from each position
    selected: list[tuple[float, SimulationTurn]] = []
    used_ids: set[str] = set()
    for pos in by_position:
        entry = by_position[pos][0]
        selected.append(entry)
        used_ids.add(entry[1].id)

    # Phase 2: fill remaining slots by novelty across all positions
    remaining = [(s, t) for s, t in scored if t.id not in used_ids]
    remaining.sort(key=lambda x: x[0], reverse=True)
    for entry in remaining:
        if len(selected) >= max_items:
            break
        selected.append(entry)

    selected.sort(key=lambda x: x[0], reverse=True)
    top = selected[:max_items]

    lines: list[str] = []
    for _, turn in top:
        persona = personas_map.get(turn.agent_persona_id)
        archetype = persona.archetype if persona else "unknown"
        position = turn.position or "neutral"
        try:
            data = json.loads(turn.content)
        except (json.JSONDecodeError, TypeError):
            data = {}
        key_point = data.get("key_point", data.get("reasoning", ""))
        if key_point:
            lines.append(f"- {archetype} ({position}): {key_point}")

    return "\n".join(lines)
