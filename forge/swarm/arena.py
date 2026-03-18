"""Arena — multi-round simulation loop with async orchestration."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from forge.swarm.consensus import ConsensusReport, extract_consensus
from forge.swarm.debate_digest import build_debate_digest
from forge.swarm.interaction import select_interactions
from forge.swarm.prompts import load_swarm_prompt

if TYPE_CHECKING:
    from collections.abc import Callable

    from forge.db.models import AgentPersona, Simulation, SimulationTurn
    from forge.db.store import Store
    from forge.swarm.population import SeedMaterial

logger = logging.getLogger(__name__)


def _agent_display_name(
    agent: AgentPersona,
    personas_map: dict[str, AgentPersona] | None = None,
) -> str:
    """Return a human-readable name for an agent persona."""
    try:
        persona = json.loads(agent.persona_json)
        name = persona.get("name", agent.archetype)
        return f"{name} ({agent.archetype})"
    except (json.JSONDecodeError, TypeError):
        return agent.archetype


def _collect_turns(
    results: list,
    population: list[AgentPersona],
    round_num: int,
    on_turn: Callable | None = None,
    personas_map: dict[str, AgentPersona] | None = None,
    diagnostics: SimulationDiagnostics | None = None,
) -> list[SimulationTurn]:
    """Filter successful turns from gather results and fire callbacks."""
    turns: list[SimulationTurn] = []
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            display = _agent_display_name(population[i])
            error_type = type(result).__name__
            logger.warning(
                "Agent %s failed in round %d: %s",
                display, round_num, error_type,
            )
            if diagnostics is not None:
                diagnostics.agent_failures.append(
                    f"Round {round_num}: {display} — {error_type}"
                )
        else:
            turns.append(result)
            if on_turn and personas_map:
                agent = personas_map.get(result.agent_persona_id)
                if agent:
                    on_turn(result, round_num, agent)
    return turns


@dataclass
class SimulationDiagnostics:
    """Tracks errors and failures during a simulation run."""

    agent_failures: list[str] = field(default_factory=list)
    population_failures: int = 0

    @property
    def total_failures(self) -> int:
        return len(self.agent_failures) + self.population_failures

    def format_summary(self) -> str:
        """Format diagnostics as a human-readable summary."""
        if self.total_failures == 0:
            return ""
        lines: list[str] = []
        if self.population_failures:
            lines.append(
                f"  Population: {self.population_failures} batch(es) failed"
            )
        round_failures: dict[int, int] = {}
        for entry in self.agent_failures:
            # entries are like "Round 1: AgentName (archetype) — ErrorType"
            try:
                r = int(entry.split(":")[0].split()[1])
                round_failures[r] = round_failures.get(r, 0) + 1
            except (IndexError, ValueError):
                pass
        for r in sorted(round_failures):
            lines.append(f"  Round {r}: {round_failures[r]} agent(s) failed")
        return "\n".join(lines)


@dataclass
class SimulationResult:
    """Result of a complete simulation run."""

    simulation: Simulation
    turns: list[SimulationTurn]
    consensus: ConsensusReport
    duration_seconds: float
    diagnostics: SimulationDiagnostics = field(default_factory=SimulationDiagnostics)


async def run_simulation(
    seed: SeedMaterial,
    population: list[AgentPersona],
    llm: object,
    store: Store,
    *,
    rounds: int = 3,
    max_concurrent: int = 2,
    on_turn: Callable | None = None,
    on_round_complete: Callable | None = None,
) -> SimulationResult:
    """Run a multi-round swarm simulation.

    Args:
        seed: The scenario seed material.
        population: List of agent personas to participate.
        llm: LLM client (LLMClient or MockLLMClient).
        store: Database store for persistence.
        rounds: Number of simulation rounds (default 3).
        max_concurrent: Max parallel LLM calls (default 2).
        on_turn: Optional callback(turn, round_num, agent) called
            after each successful agent turn for real-time display.

    Returns:
        SimulationResult with simulation record, turns, and consensus.
    """
    start = time.monotonic()
    sem = asyncio.Semaphore(max_concurrent)
    diagnostics = SimulationDiagnostics()

    # Create simulation record
    sim = store.save_simulation(
        mode="scenario",
        seed_text=seed.text,
        seed_context=seed.context,
        agent_count=len(population),
        rounds=rounds,
    )
    store.update_simulation(sim.id, status="running")

    all_turns: list[SimulationTurn] = []
    personas_map = {p.id: p for p in population}

    try:
        # Round 1: Initial reactions
        r1_turns = await _run_round1(
            seed, population, llm, store, sim.id, sem,
            on_turn=on_turn, personas_map=personas_map,
            diagnostics=diagnostics,
        )
        all_turns.extend(r1_turns)
        if on_round_complete:
            on_round_complete(1, r1_turns, population)

        # Round 2: Interactions (if rounds >= 2)
        r2_turns: list[SimulationTurn] = []
        if rounds >= 2:
            r2_turns = await _run_round2(
                seed, population, r1_turns, personas_map, llm, store, sim.id, sem,
                on_turn=on_turn, diagnostics=diagnostics,
            )
            all_turns.extend(r2_turns)
            if on_round_complete:
                on_round_complete(2, r2_turns, population, r1_turns)

        # Round 3: Convergence (if rounds >= 3)
        if rounds >= 3:
            r3_turns = await _run_round3(
                seed, population, r1_turns, r2_turns, llm, store, sim.id, sem,
                on_turn=on_turn, personas_map=personas_map,
                diagnostics=diagnostics,
            )
            all_turns.extend(r3_turns)
            if on_round_complete:
                on_round_complete(3, r3_turns, population, r2_turns)

        # Extract consensus
        consensus = extract_consensus(all_turns, personas_map)

        duration = time.monotonic() - start
        store.update_simulation(
            sim.id,
            status="complete",
            duration_seconds=round(duration, 2),
        )
        sim = store.get_simulation(sim.id)  # type: ignore[assignment]

        return SimulationResult(
            simulation=sim,  # type: ignore[arg-type]
            turns=all_turns,
            consensus=consensus,
            duration_seconds=round(duration, 2),
            diagnostics=diagnostics,
        )

    except Exception:
        store.update_simulation(sim.id, status="failed")
        raise


async def _run_round1(
    seed: SeedMaterial,
    population: list[AgentPersona],
    llm: object,
    store: Store,
    sim_id: str,
    sem: asyncio.Semaphore,
    *,
    on_turn: Callable | None = None,
    personas_map: dict[str, AgentPersona] | None = None,
    diagnostics: SimulationDiagnostics | None = None,
) -> list[SimulationTurn]:
    """Round 1: Each agent independently reacts to the scenario."""

    async def react(agent: AgentPersona) -> SimulationTurn:
        persona = json.loads(agent.persona_json)
        personality = persona.get("personality", {})
        prompt = load_swarm_prompt(
            "reaction",
            agent_name=persona.get("name", agent.archetype),
            agent_background=persona.get("background", ""),
            agent_archetype=agent.archetype,
            agent_expertise=", ".join(persona.get("expertise", [])),
            risk_appetite=personality.get("risk_appetite", "medium"),
            optimism_bias=personality.get("optimism_bias", "realist"),
            reasoning_style=persona.get("reasoning_style", "analytical"),
            seed_text=seed.text,
            seed_context=seed.context or "",
            confidence_anchor=persona.get("confidence_anchor", "medium (46-70)"),
        )
        async with sem:
            response = await llm.complete(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.8,
            )
        data = response.parsed_json or {}
        return store.save_simulation_turn(
            simulation_id=sim_id,
            round=1,
            agent_persona_id=agent.id,
            turn_type="reaction",
            content=json.dumps(data),
            position=data.get("position", "neutral"),
            confidence=data.get("confidence", 50),
            token_count=response.token_count,
            raw_content=response.content,
        )

    tasks = [react(agent) for agent in population]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return _collect_turns(results, population, 1, on_turn, personas_map, diagnostics)


async def _run_round2(
    seed: SeedMaterial,
    population: list[AgentPersona],
    r1_turns: list[SimulationTurn],
    personas_map: dict[str, AgentPersona],
    llm: object,
    store: Store,
    sim_id: str,
    sem: asyncio.Semaphore,
    *,
    on_turn: Callable | None = None,
    diagnostics: SimulationDiagnostics | None = None,
) -> list[SimulationTurn]:
    """Round 2: Each agent responds to selected opposing views."""
    # Build lookup of r1 turns by agent
    r1_by_agent = {t.agent_persona_id: t for t in r1_turns}

    async def interact(agent: AgentPersona) -> SimulationTurn:
        my_turn = r1_by_agent.get(agent.id)
        if my_turn is None:
            # Agent didn't participate in round 1 — skip
            return store.save_simulation_turn(
                simulation_id=sim_id, round=2, agent_persona_id=agent.id,
                turn_type="reaction", content='{"position":"neutral","confidence":50}',
                position="neutral", confidence=50,
            )

        opposing = select_interactions(agent, my_turn, r1_turns, personas_map, count=3)

        # Format opposing views for prompt
        opposing_views = []
        for opp_turn in opposing:
            opp_persona = personas_map.get(opp_turn.agent_persona_id)
            opp_data = _safe_json(opp_turn.content)
            opposing_views.append({
                "archetype": opp_persona.archetype if opp_persona else "unknown",
                "position": opp_data.get("position", "neutral"),
                "confidence": opp_data.get("confidence", 50),
                "reasoning": opp_data.get("reasoning", ""),
                "key_concern": opp_data.get("key_concern", ""),
            })

        my_data = _safe_json(my_turn.content)
        persona = json.loads(agent.persona_json)
        prompt = load_swarm_prompt(
            "interaction",
            agent_name=persona.get("name", agent.archetype),
            agent_background=persona.get("background", ""),
            agent_archetype=agent.archetype,
            reasoning_style=persona.get("reasoning_style", "analytical"),
            seed_text=seed.text,
            my_position=my_data.get("position", "neutral"),
            my_confidence=str(my_data.get("confidence", 50)),
            my_reasoning=my_data.get("reasoning", ""),
            opposing_views=opposing_views,
        )

        async with sem:
            response = await llm.complete(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.8,
            )

        data = response.parsed_json or {}
        responding_to = opposing[0].id if opposing else None
        return store.save_simulation_turn(
            simulation_id=sim_id,
            round=2,
            agent_persona_id=agent.id,
            turn_type=data.get("turn_type", "challenge"),
            content=json.dumps(data),
            responding_to_id=responding_to,
            position=data.get("position", "neutral"),
            confidence=data.get("confidence", 50),
            token_count=response.token_count,
            raw_content=response.content,
        )

    tasks = [interact(agent) for agent in population]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return _collect_turns(results, population, 2, on_turn, personas_map, diagnostics)


async def _run_round3(
    seed: SeedMaterial,
    population: list[AgentPersona],
    r1_turns: list[SimulationTurn],
    r2_turns: list[SimulationTurn],
    llm: object,
    store: Store,
    sim_id: str,
    sem: asyncio.Semaphore,
    *,
    on_turn: Callable | None = None,
    personas_map: dict[str, AgentPersona] | None = None,
    diagnostics: SimulationDiagnostics | None = None,
) -> list[SimulationTurn]:
    """Round 3: Final positions with conviction deltas."""
    r1_by_agent = {t.agent_persona_id: t for t in r1_turns}
    r2_by_agent = {t.agent_persona_id: t for t in r2_turns}

    async def converge(agent: AgentPersona) -> SimulationTurn:
        r1_turn = r1_by_agent.get(agent.id)
        r2_turn = r2_by_agent.get(agent.id)

        r1_data = _safe_json(r1_turn.content) if r1_turn else {}
        r2_data = _safe_json(r2_turn.content) if r2_turn else {}

        persona = json.loads(agent.persona_json)
        digest = build_debate_digest(
            r2_turns, agent.id, personas_map or {},
        )
        prompt = load_swarm_prompt(
            "convergence",
            agent_name=persona.get("name", agent.archetype),
            agent_background=persona.get("background", ""),
            seed_text=seed.text,
            r1_position=r1_data.get("position", "neutral"),
            r1_confidence=str(r1_data.get("confidence", 50)),
            r1_reasoning=r1_data.get("reasoning", ""),
            r2_summary=r2_data.get("reasoning", "No interaction recorded."),
            debate_digest=digest,
        )

        async with sem:
            response = await llm.complete(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.7,
            )

        data = response.parsed_json or {}
        position = data.get("final_position", data.get("position", "neutral"))
        return store.save_simulation_turn(
            simulation_id=sim_id,
            round=3,
            agent_persona_id=agent.id,
            turn_type="convergence",
            content=json.dumps(data),
            position=position,
            confidence=data.get("confidence", 50),
            token_count=response.token_count,
            raw_content=response.content,
        )

    tasks = [converge(agent) for agent in population]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return _collect_turns(results, population, 3, on_turn, personas_map, diagnostics)


def _safe_json(content: str) -> dict:
    """Safely parse JSON content, returning empty dict on failure."""
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {}
