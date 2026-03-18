"""Interaction selection — pick opposing views for Round 2 debate."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.db.models import AgentPersona, SimulationTurn

logger = logging.getLogger(__name__)

# Words too common to signal novelty — stop words + domain-generic terms
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "this", "that", "it", "not", "no",
    "as", "if", "its", "their", "which", "while", "also", "both", "more",
    "than", "such", "these", "those", "into", "over", "all", "any",
    # Domain-generic terms that add no signal
    "regulation", "innovation", "framework", "approach", "balance",
    "risk", "ensure", "require", "without", "between", "through",
    "must", "need", "like", "new", "way", "based", "well",
})

_WORD_RE = re.compile(r"[a-z]{3,}")


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


def _extract_words(turn: SimulationTurn) -> list[str]:
    """Extract meaningful words from a turn's reasoning and key_concern."""
    try:
        data = json.loads(turn.content)
    except (json.JSONDecodeError, TypeError):
        return []
    text = data.get("reasoning", "") + " " + data.get("key_concern", "")
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOP_WORDS]


def compute_novelty_score(
    turn: SimulationTurn,
    all_turns: list[SimulationTurn],
) -> float:
    """Score how unique a turn's reasoning is compared to all other turns.

    Returns a float 0.0-1.0 where higher means more novel vocabulary.
    """
    my_words = _extract_words(turn)
    if not my_words:
        return 0.0

    # Count how many other turns contain each word
    other_turns = [t for t in all_turns if t.id != turn.id]
    if not other_turns:
        return 1.0

    word_doc_count: Counter[str] = Counter()
    for other in other_turns:
        other_words = set(_extract_words(other))
        for w in other_words:
            word_doc_count[w] += 1

    threshold = len(other_turns) * 0.2
    rare_count = sum(1 for w in my_words if word_doc_count[w] < threshold)
    return rare_count / len(my_words)


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
    2. Most novel opposing argument (highest novelty score)
    3. Domain expert disagreement (agent with expertise who disagrees)

    Falls back to highest contrarian_tendency regardless of position
    when no opposing views exist.
    """
    if not all_round1_turns:
        return []

    my_position = agent_turn.position
    other_turns = [t for t in all_round1_turns if t.agent_persona_id != agent.id]

    if not other_turns:
        return []

    opposing = [t for t in other_turns if t.position != my_position]

    if not opposing:
        return _select_by_contrarian(other_turns, all_personas, count)

    selected: list[SimulationTurn] = []
    used_ids: set[str] = set()

    # 1. Strongest opposing argument (highest confidence)
    strongest = max(opposing, key=lambda t: t.confidence or 0)
    selected.append(strongest)
    used_ids.add(strongest.id)

    if len(selected) >= count:
        return selected[:count]

    # 2. Most novel opposing argument — biased toward least-common position
    #    when the agent is in the majority, to prevent minority crowding-out
    remaining = [t for t in opposing if t.id not in used_ids]
    if remaining:
        novelty_candidates = remaining
        if _is_majority_position(my_position, all_round1_turns):
            least_common = _least_common_position(remaining)
            if least_common:
                lc_candidates = [t for t in remaining if t.position == least_common]
                if lc_candidates:
                    novelty_candidates = lc_candidates
        most_novel = max(
            novelty_candidates,
            key=lambda t: compute_novelty_score(t, all_round1_turns),
        )
        selected.append(most_novel)
        used_ids.add(most_novel.id)

    if len(selected) >= count:
        return selected[:count]

    # 3. Domain expert disagreement
    remaining = [t for t in opposing if t.id not in used_ids]
    if remaining:
        expert = _find_domain_expert(remaining, all_personas)
        if expert:
            selected.append(expert)
            used_ids.add(expert.id)

    if len(selected) >= count:
        return selected[:count]

    # Fill remaining slots from unused opposing turns
    remaining = [t for t in opposing if t.id not in used_ids]
    for turn in remaining:
        if len(selected) >= count:
            break
        selected.append(turn)

    return selected[:count]


def _is_majority_position(
    position: str,
    all_turns: list[SimulationTurn],
) -> bool:
    """Check if a position is held by >50% of agents."""
    if not all_turns:
        return False
    count = sum(1 for t in all_turns if t.position == position)
    return count > len(all_turns) / 2


def _least_common_position(turns: list[SimulationTurn]) -> str | None:
    """Return the least-common position among a set of turns."""
    if not turns:
        return None
    counts: Counter[str] = Counter(t.position for t in turns)
    return counts.most_common()[-1][0]


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
