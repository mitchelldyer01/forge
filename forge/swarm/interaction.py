"""Interaction selection — pick opposing views for Round 2 debate."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.db.models import AgentPersona, SimulationTurn

logger = logging.getLogger(__name__)


def _parse_persona(persona_json: str) -> dict:
    """Parse persona_json string into a dict."""
    try:
        return json.loads(persona_json)
    except (json.JSONDecodeError, TypeError):
        return {}


def _get_contrarian_tendency(persona: dict) -> float:
    """Extract contrarian_tendency from persona personality."""
    personality = persona.get("personality", {})
    return float(personality.get("contrarian_tendency", 0.0))


def _get_expertise(persona: dict) -> list[str]:
    """Extract expertise list from persona."""
    return persona.get("expertise", [])


def select_interactions(
    agent: AgentPersona,
    agent_turn: SimulationTurn,
    all_round1_turns: list[SimulationTurn],
    all_personas: dict[str, AgentPersona],
    count: int = 3,
) -> list[SimulationTurn]:
    """Select opposing views for an agent to respond to in Round 2.

    Selection priority:
    1. Strongest opposing argument (highest confidence from other side)
    2. Most contrarian perspective (highest contrarian_tendency on other side)
    3. Domain expert disagreement (agent with expertise who disagrees)

    Falls back to highest contrarian_tendency regardless of position
    when no opposing views exist.

    Args:
        agent: The agent who will respond.
        agent_turn: The agent's Round 1 turn.
        all_round1_turns: All Round 1 turns from all agents.
        all_personas: Dict mapping persona ID to AgentPersona.
        count: Maximum number of interactions to select.

    Returns:
        List of SimulationTurns for the agent to respond to.
    """
    if not all_round1_turns:
        return []

    my_position = agent_turn.position
    # Exclude self
    other_turns = [t for t in all_round1_turns if t.agent_persona_id != agent.id]

    if not other_turns:
        return []

    # Split into opposing and same-position
    opposing = [t for t in other_turns if t.position != my_position]

    if not opposing:
        # Fallback: no opposing views, pick most contrarian regardless
        return _select_by_contrarian(other_turns, all_personas, count)

    selected: list[SimulationTurn] = []
    used_ids: set[str] = set()

    # 1. Strongest opposing argument (highest confidence)
    strongest = max(opposing, key=lambda t: t.confidence or 0)
    selected.append(strongest)
    used_ids.add(strongest.id)

    if len(selected) >= count:
        return selected[:count]

    # 2. Most contrarian opposing agent
    remaining_opposing = [t for t in opposing if t.id not in used_ids]
    if remaining_opposing:
        most_contrarian = max(
            remaining_opposing,
            key=lambda t: _get_contrarian_tendency(
                _parse_persona(all_personas[t.agent_persona_id].persona_json)
            ) if t.agent_persona_id in all_personas else 0.0,
        )
        selected.append(most_contrarian)
        used_ids.add(most_contrarian.id)

    if len(selected) >= count:
        return selected[:count]

    # 3. Domain expert disagreement
    remaining_opposing = [t for t in opposing if t.id not in used_ids]
    if remaining_opposing:
        expert = _find_domain_expert(remaining_opposing, all_personas)
        if expert:
            selected.append(expert)
            used_ids.add(expert.id)

    if len(selected) >= count:
        return selected[:count]

    # Fill remaining slots from any unused opposing turns
    remaining_opposing = [t for t in opposing if t.id not in used_ids]
    for turn in remaining_opposing:
        if len(selected) >= count:
            break
        selected.append(turn)

    return selected[:count]


def _select_by_contrarian(
    turns: list[SimulationTurn],
    all_personas: dict[str, AgentPersona],
    count: int,
) -> list[SimulationTurn]:
    """Fallback: select by contrarian tendency when no opposing views exist."""
    scored = []
    for turn in turns:
        persona = all_personas.get(turn.agent_persona_id)
        ct = (
            _get_contrarian_tendency(_parse_persona(persona.persona_json))
            if persona
            else 0.0
        )
        scored.append((ct, turn))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [turn for _, turn in scored[:count]]


def _find_domain_expert(
    turns: list[SimulationTurn],
    all_personas: dict[str, AgentPersona],
) -> SimulationTurn | None:
    """Find a turn from an agent with non-empty expertise."""
    for turn in turns:
        persona = all_personas.get(turn.agent_persona_id)
        if persona:
            expertise = _get_expertise(_parse_persona(persona.persona_json))
            if expertise:
                return turn
    return None
