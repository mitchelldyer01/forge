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
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:max_items]

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
