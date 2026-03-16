"""Consensus extraction — pure computation on simulation turns."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.db.models import AgentPersona, SimulationTurn

logger = logging.getLogger(__name__)


@dataclass
class DissentCluster:
    """A group of agents who disagree with the majority."""

    position: str
    agent_count: int
    avg_confidence: float
    key_arguments: list[str] = field(default_factory=list)


@dataclass
class ConvictionShift:
    """An agent who changed position between round 1 and round 3."""

    agent_persona_id: str
    archetype: str
    from_position: str
    to_position: str
    confidence_delta: int
    reason: str = ""


@dataclass
class EdgeCase:
    """A unique perspective held by only 1-2 agents."""

    agent_persona_id: str
    archetype: str
    position: str
    reasoning: str


@dataclass
class PredictionCandidate:
    """A potential prediction extracted from simulation output."""

    claim: str
    supporting_agent_count: int
    avg_confidence: float
    includes_domain_experts: bool = False
    includes_shifted_agents: bool = False


@dataclass
class ConsensusReport:
    """Full consensus analysis from a simulation."""

    majority_position: str
    majority_confidence: float
    majority_fraction: float
    dissent_clusters: list[DissentCluster] = field(default_factory=list)
    conviction_shifts: list[ConvictionShift] = field(default_factory=list)
    edge_cases: list[EdgeCase] = field(default_factory=list)
    prediction_candidates: list[PredictionCandidate] = field(default_factory=list)


def extract_consensus(
    turns: list[SimulationTurn],
    personas: dict[str, AgentPersona],
) -> ConsensusReport:
    """Extract consensus from simulation turns. Pure computation, no LLM.

    Args:
        turns: All simulation turns (rounds 1-3).
        personas: Dict mapping persona ID to AgentPersona.

    Returns:
        ConsensusReport with majority, dissent, shifts, and edge cases.
    """
    if not turns:
        return ConsensusReport(
            majority_position="neutral",
            majority_confidence=0.0,
            majority_fraction=0.0,
        )

    round3 = [t for t in turns if t.round == 3]
    round1 = [t for t in turns if t.round == 1]

    # If no round 3 turns, use whatever rounds exist
    if not round3:
        round3 = turns

    # Group round 3 by position
    position_groups: dict[str, list[SimulationTurn]] = defaultdict(list)
    for turn in round3:
        pos = turn.position or "neutral"
        position_groups[pos].append(turn)

    # Find majority (tie-break by average confidence)
    majority_pos, majority_turns = _find_majority(position_groups)
    total_agents = len(round3)
    majority_confidence = _avg_confidence(majority_turns)
    majority_fraction = len(majority_turns) / total_agents if total_agents else 0.0

    # Dissent clusters
    dissent_clusters = _build_dissent_clusters(position_groups, majority_pos)

    # Conviction shifts (round 1 vs round 3)
    conviction_shifts = _find_conviction_shifts(round1, round3, personas)

    # Edge cases (positions held by <= 2 agents, excluding majority)
    edge_cases = _find_edge_cases(position_groups, majority_pos, personas)

    return ConsensusReport(
        majority_position=majority_pos,
        majority_confidence=majority_confidence,
        majority_fraction=majority_fraction,
        dissent_clusters=dissent_clusters,
        conviction_shifts=conviction_shifts,
        edge_cases=edge_cases,
    )


def _find_majority(
    groups: dict[str, list[SimulationTurn]],
) -> tuple[str, list[SimulationTurn]]:
    """Find majority position. Tie-break by highest average confidence."""
    if not groups:
        return "neutral", []

    max_count = max(len(g) for g in groups.values())
    candidates = [(pos, g) for pos, g in groups.items() if len(g) == max_count]

    if len(candidates) == 1:
        return candidates[0]

    # Tie-break by average confidence
    return max(candidates, key=lambda x: _avg_confidence(x[1]))


def _avg_confidence(turns: list[SimulationTurn]) -> float:
    """Compute average confidence from turns."""
    confidences = [t.confidence for t in turns if t.confidence is not None]
    if not confidences:
        return 0.0
    return sum(confidences) / len(confidences)


def _build_dissent_clusters(
    groups: dict[str, list[SimulationTurn]],
    majority_pos: str,
) -> list[DissentCluster]:
    """Build dissent clusters from non-majority positions."""
    clusters = []
    for pos, group_turns in groups.items():
        if pos == majority_pos:
            continue
        arguments = _extract_arguments(group_turns)
        clusters.append(DissentCluster(
            position=pos,
            agent_count=len(group_turns),
            avg_confidence=_avg_confidence(group_turns),
            key_arguments=arguments,
        ))
    return clusters


def _extract_arguments(turns: list[SimulationTurn]) -> list[str]:
    """Extract reasoning from turn content JSON."""
    arguments = []
    for turn in turns:
        try:
            data = json.loads(turn.content)
            reasoning = data.get("reasoning", "")
            if reasoning:
                arguments.append(reasoning)
        except (json.JSONDecodeError, TypeError):
            pass
    return arguments


def _find_conviction_shifts(
    round1: list[SimulationTurn],
    round3: list[SimulationTurn],
    personas: dict[str, AgentPersona],
) -> list[ConvictionShift]:
    """Find agents who changed position between rounds."""
    r1_by_agent = {t.agent_persona_id: t for t in round1}
    shifts = []
    for r3_turn in round3:
        r1_turn = r1_by_agent.get(r3_turn.agent_persona_id)
        if r1_turn is None:
            continue
        if r1_turn.position != r3_turn.position:
            persona = personas.get(r3_turn.agent_persona_id)
            archetype = persona.archetype if persona else "unknown"
            r1_conf = r1_turn.confidence or 0
            r3_conf = r3_turn.confidence or 0
            shifts.append(ConvictionShift(
                agent_persona_id=r3_turn.agent_persona_id,
                archetype=archetype,
                from_position=r1_turn.position or "neutral",
                to_position=r3_turn.position or "neutral",
                confidence_delta=r3_conf - r1_conf,
            ))
    return shifts


def _find_edge_cases(
    groups: dict[str, list[SimulationTurn]],
    majority_pos: str,
    personas: dict[str, AgentPersona],
) -> list[EdgeCase]:
    """Find positions held by only 1-2 agents (excluding majority)."""
    edge_cases = []
    for pos, group_turns in groups.items():
        if pos == majority_pos:
            continue
        if len(group_turns) <= 2:
            for turn in group_turns:
                persona = personas.get(turn.agent_persona_id)
                archetype = persona.archetype if persona else "unknown"
                reasoning = ""
                try:
                    data = json.loads(turn.content)
                    reasoning = data.get("reasoning", "")
                except (json.JSONDecodeError, TypeError):
                    pass
                edge_cases.append(EdgeCase(
                    agent_persona_id=turn.agent_persona_id,
                    archetype=archetype,
                    position=pos,
                    reasoning=reasoning,
                ))
    return edge_cases
