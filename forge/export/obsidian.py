"""One-way Obsidian vault renderer from SQLite."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.db.models import Hypothesis, Prediction, Simulation
    from forge.db.store import Store


def _sanitize_filename(text: str, max_len: int = 60) -> str:
    """Convert text to a safe filename."""
    safe = re.sub(r"[^\w\s-]", "", text)
    safe = re.sub(r"\s+", "_", safe.strip())
    return safe[:max_len]


def _render_hypothesis(h: Hypothesis, relations_text: str) -> str:
    """Render a hypothesis as Obsidian markdown."""
    tags = " ".join(f"#{t}" for t in (h.tags or []))
    lines = [
        f"# {h.claim}",
        "",
        f"**ID:** `{h.id}`",
        f"**Status:** {h.status}",
        f"**Confidence:** {h.confidence}/100",
        f"**Source:** {h.source}",
        f"**Created:** {h.created_at}",
        f"**Updated:** {h.updated_at}",
    ]
    if h.context:
        lines.extend(["", "## Context", "", h.context])
    if h.parent_id:
        lines.append(f"**Parent:** `{h.parent_id}`")
    if h.resolution_deadline:
        lines.append(f"**Resolution deadline:** {h.resolution_deadline}")
    lines.extend([
        "",
        f"**Challenges survived:** {h.challenges_survived}",
        f"**Challenges failed:** {h.challenges_failed}",
        f"**Human endorsed:** {h.human_endorsed}",
        f"**Human rejected:** {h.human_rejected}",
    ])
    if tags:
        lines.extend(["", f"**Tags:** {tags}"])
    if relations_text:
        lines.extend(["", "## Relations", "", relations_text])
    return "\n".join(lines) + "\n"


def _render_prediction(p: Prediction) -> str:
    """Render a prediction as Obsidian markdown."""
    lines = [
        f"# {p.claim}",
        "",
        f"**ID:** `{p.id}`",
        f"**Confidence:** {p.confidence}/100",
        f"**Simulation:** `{p.simulation_id}`",
    ]
    if p.consensus_strength is not None:
        lines.append(f"**Consensus strength:** {p.consensus_strength:.1%}")
    if p.resolution_deadline:
        lines.append(f"**Resolution deadline:** {p.resolution_deadline}")
    if p.resolved_as:
        lines.append(f"**Resolved as:** {p.resolved_as}")
        if p.resolved_at:
            lines.append(f"**Resolved at:** {p.resolved_at}")
    if p.dissent_summary:
        lines.extend(["", "## Dissent", "", p.dissent_summary])
    return "\n".join(lines) + "\n"


def _render_simulation(s: Simulation) -> str:
    """Render a simulation as Obsidian markdown."""
    lines = [
        f"# {s.seed_text}",
        "",
        f"**ID:** `{s.id}`",
        f"**Mode:** {s.mode}",
        f"**Status:** {s.status}",
    ]
    if s.agent_count:
        lines.append(f"**Agents:** {s.agent_count}")
    if s.rounds:
        lines.append(f"**Rounds:** {s.rounds}")
    if s.started_at:
        lines.append(f"**Started:** {s.started_at}")
    if s.completed_at:
        lines.append(f"**Completed:** {s.completed_at}")
    if s.duration_seconds:
        lines.append(f"**Duration:** {s.duration_seconds:.1f}s")
    if s.summary:
        lines.extend(["", "## Summary", "", s.summary])
    return "\n".join(lines) + "\n"


def render_vault(store: Store, vault_path: str) -> None:
    """Render the full Obsidian vault from current DB state.

    One-way sync: always overwrites existing files.
    """
    base = Path(vault_path)
    hyp_dir = base / "hypotheses"
    pred_dir = base / "predictions"
    sim_dir = base / "simulations"

    for d in (hyp_dir, pred_dir, sim_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Clean existing files
    for d in (hyp_dir, pred_dir, sim_dir):
        for f in d.glob("*.md"):
            f.unlink()

    # Render hypotheses
    hypotheses = store.list_hypotheses()
    for h in hypotheses:
        relations = store.list_relations_for_hypothesis(h.id)
        rel_lines = []
        for r in relations:
            if r.source_id == h.id:
                rel_lines.append(
                    f"- **{r.relation_type}** → `{r.target_id}` "
                    f"(strength: {r.strength})"
                )
            else:
                rel_lines.append(
                    f"- **{r.relation_type}** ← `{r.source_id}` "
                    f"(strength: {r.strength})"
                )
        relations_text = "\n".join(rel_lines)

        filename = f"{_sanitize_filename(h.claim)}.md"
        (hyp_dir / filename).write_text(_render_hypothesis(h, relations_text))

    # Render predictions
    predictions = store.list_predictions()
    for p in predictions:
        filename = f"{_sanitize_filename(p.claim)}.md"
        (pred_dir / filename).write_text(_render_prediction(p))

    # Render simulations
    simulations = store.list_simulations()
    for s in simulations:
        filename = f"{_sanitize_filename(s.seed_text)}.md"
        (sim_dir / filename).write_text(_render_simulation(s))
