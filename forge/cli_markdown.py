"""Markdown rendering for simulation turns — LLM-friendly output."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.db.models import Prediction, Simulation

ROUND_LABELS = {1: "Initial Reactions", 2: "Debate", 3: "Final Positions"}


def _safe_parse(content: str | None) -> dict:
    """Parse JSON content, returning empty dict on failure."""
    if not content:
        return {}
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {}


def _render_header(sim: Simulation) -> str:
    """Render simulation metadata header."""
    lines = [
        f"# Simulation: {sim.seed_text}",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| ID | {sim.id} |",
        f"| Agents | {sim.agent_count or '-'} |",
        f"| Rounds | {sim.rounds or '-'} |",
        f"| Status | {sim.status} |",
    ]
    if sim.started_at:
        lines.append(f"| Started | {sim.started_at} |")
    return "\n".join(lines)


def _render_turn(row: dict, round_num: int) -> str:
    """Render a single turn as markdown."""
    content = _safe_parse(row.get("content"))
    archetype = row.get("archetype", "unknown")
    position = row.get("position") or content.get("position", "unknown")
    confidence = row.get("confidence")
    conf_str = f"{confidence}%" if confidence is not None else "?"
    turn_type = row.get("turn_type", "")

    parts = [f"### {archetype} — {position} ({conf_str})"]

    if round_num == 2 and turn_type:
        parts[0] += f" [{turn_type}]"

    if round_num == 3:
        delta = content.get("conviction_delta")
        changed = content.get("changed_mind")
        meta = []
        if delta is not None:
            sign = "+" if delta >= 0 else ""
            meta.append(f"**Conviction delta:** {sign}{delta}")
        if changed is not None:
            meta.append(f"**Changed mind:** {'Yes' if changed else 'No'}")
        if meta:
            parts.append("")
            parts.append(" | ".join(meta))

    reasoning = content.get("reasoning", "")
    if reasoning:
        parts.append("")
        parts.append(reasoning)

    quotes = []
    if round_num == 1:
        for field in ("key_concern", "key_insight"):
            val = content.get(field)
            if val:
                label = field.replace("_", " ").title()
                quotes.append(f"> **{label}:** {val}")
    elif round_num == 2:
        steel = content.get("steel_man")
        if steel:
            quotes.append(f"> **Steel-man:** {steel}")
        mechanism = content.get("concrete_mechanism")
        if mechanism and mechanism.lower() != "none":
            quotes.append(f"> **Mechanism:** {mechanism}")
        val = content.get("key_point")
        if val:
            quotes.append(f"> **Key point:** {val}")
    elif round_num == 3:
        justification = content.get("confidence_justification")
        if justification:
            quotes.append(f"> **Confidence basis:** {justification}")
        val = content.get("key_insight")
        if val:
            quotes.append(f"> **Key insight:** {val}")

    if quotes:
        parts.append("")
        parts.extend(quotes)

    return "\n".join(parts)


def _render_consensus(rows: list[dict]) -> str:
    """Compute and render a lightweight consensus from round 3 turns."""
    r3 = [r for r in rows if r["round"] == 3]
    if not r3:
        return ""

    pos_counts: dict[str, list[int]] = defaultdict(list)
    for row in r3:
        pos = row.get("position") or "neutral"
        pos_counts[pos].append(row.get("confidence") or 0)

    majority_pos = max(pos_counts, key=lambda p: len(pos_counts[p]))
    majority_confs = pos_counts[majority_pos]
    majority_avg = sum(majority_confs) / len(majority_confs)
    total = len(r3)

    lines = [
        "## Consensus",
        "",
        f"**Majority:** {majority_pos} "
        f"({majority_avg:.0f}% confidence, "
        f"held by {len(majority_confs)}/{total} agents)",
    ]

    # Confidence trend across rounds
    trend_parts = []
    for rnd in sorted({r["round"] for r in rows}):
        confs = [r.get("confidence", 0) for r in rows if r["round"] == rnd]
        if confs:
            trend_parts.append(f"{sum(confs) / len(confs):.0f}%")
    if len(trend_parts) > 1:
        lines.append(f"**Confidence trend:** {' -> '.join(trend_parts)}")

    # Dissent
    dissent = {p: c for p, c in pos_counts.items() if p != majority_pos}
    if dissent:
        lines.append("")
        lines.append("### Dissent")
        for pos, confs in dissent.items():
            avg = sum(confs) / len(confs)
            lines.append(f"- {pos}: {len(confs)} agent(s) (avg {avg:.0f}%)")

    # Conviction shifts (R1 vs R3)
    r1_positions = {
        r["agent_persona_id"]: r.get("position")
        for r in rows if r["round"] == 1
    }
    shifts = []
    for row in r3:
        aid = row["agent_persona_id"]
        r1_pos = r1_positions.get(aid)
        r3_pos = row.get("position")
        if r1_pos and r3_pos and r1_pos != r3_pos:
            content = _safe_parse(row.get("content"))
            delta = content.get("conviction_delta", "?")
            shifts.append(
                f"- {row['archetype']}: {r1_pos} -> {r3_pos} (delta: {delta})"
            )
    if shifts:
        lines.append("")
        lines.append("### Conviction Shifts")
        lines.extend(shifts)

    return "\n".join(lines)


def _render_scenario(sim: Simulation) -> str:
    """Render the scenario seed material."""
    lines = ["## Scenario", "", sim.seed_text]
    if sim.seed_context:
        lines.append("")
        lines.append(f"**Context:** {sim.seed_context}")
    return "\n".join(lines)


def _render_agent_roster(rows: list[dict]) -> str:
    """Render unique agent personas from the turn rows."""
    seen: dict[str, dict] = {}
    for row in rows:
        archetype = row.get("archetype", "unknown")
        if archetype in seen:
            continue
        persona = _safe_parse(row.get("persona_json"))
        seen[archetype] = persona

    if not seen:
        return ""

    lines = ["## Agent Roster", ""]
    for archetype, persona in seen.items():
        parts = [f"**{archetype}**"]
        name = persona.get("name")
        if name:
            parts[0] += f" ({name})"
        traits = []
        bg = persona.get("background")
        if bg:
            traits.append(f"Background: {bg}")
        expertise = persona.get("expertise")
        if expertise and isinstance(expertise, list):
            traits.append(f"Expertise: {', '.join(expertise)}")
        style = persona.get("reasoning_style")
        if style:
            traits.append(f"Style: {style}")
        if traits:
            parts.append("; ".join(traits))
        lines.append("- " + " — ".join(parts))

    return "\n".join(lines)


def _render_predictions(predictions: list[Prediction]) -> str:
    """Render predictions extracted from the simulation."""
    if not predictions:
        return ""

    lines = ["## Predictions", ""]
    for p in predictions:
        status = p.resolved_as or "pending"
        deadline = p.resolution_deadline or "no deadline"
        lines.append(
            f"- **{p.claim}** ({p.confidence}% confidence, "
            f"deadline: {deadline}, status: {status})"
        )
    return "\n".join(lines)


def render_turns_markdown(
    rows: list[dict],
    sim: Simulation,
    *,
    predictions: list[Prediction] | None = None,
) -> str:
    """Render simulation turns as structured markdown for LLM analysis."""
    sections = [_render_header(sim)]
    sections.append(_render_scenario(sim))

    if not rows:
        return "\n\n".join(s for s in sections if s)

    sections.append(_render_agent_roster(rows))

    # Group by round
    by_round: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_round[row["round"]].append(row)

    for rnd in sorted(by_round):
        label = ROUND_LABELS.get(rnd, f"Round {rnd}")
        sections.append(f"## Round {rnd} — {label}")
        for row in by_round[rnd]:
            sections.append(_render_turn(row, rnd))

    sections.append(_render_consensus(rows))

    if predictions:
        sections.append(_render_predictions(predictions))

    return "\n\n".join(s for s in sections if s)
