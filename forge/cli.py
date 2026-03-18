"""CLI entrypoint for FORGE — Typer app."""

from __future__ import annotations

import asyncio
import json
import logging

import httpx
import typer
from rich.console import Console
from rich.panel import Panel

from forge.analyze.structured import AnalysisError, analyze
from forge.config import Settings
from forge.db.store import Store
from forge.llm.client import LLMClient
from forge.swarm.population import SeedMaterial

app = typer.Typer(
    name="forge",
    help="FORGE — Calibrated Prediction Engine",
    invoke_without_command=True,
)
feed_app = typer.Typer(help="Manage RSS feeds")
app.add_typer(feed_app, name="feed")
console = Console()

# Show LLM client warnings (malformed JSON retries, 503 retries) in terminal
logging.basicConfig(format="%(message)s", level=logging.WARNING)


@app.callback()
def main() -> None:
    """FORGE — Calibrated Prediction Engine."""


def _format_error(e: Exception) -> str:
    """Format an exception into a helpful terminal error message."""
    if isinstance(e, AnalysisError):
        lines = [f"[red]Error:[/red] Analysis failed at [bold]{e.stage}[/bold] stage"]
        lines.append(f"  Cause: {e.original_error}")
        return "\n".join(lines)
    if isinstance(e, httpx.ConnectError):
        return (
            f"[red]Error:[/red] Cannot connect to LLM server: {e}\n"
            "[dim]Hint: Is llama-server running? Check `forge status`.[/dim]"
        )
    if isinstance(e, httpx.HTTPStatusError):
        return f"[red]Error:[/red] {e}"
    # Generic fallback — include exception type for debuggability
    return f"[red]Error:[/red] {type(e).__name__}: {e}"


def _get_store() -> Store:
    settings = Settings()
    return Store(settings.db_path)


def _get_llm() -> LLMClient:
    settings = Settings()
    return LLMClient(base_url=settings.llama_url, timeout=settings.llama_timeout)


@app.command()
def test(
    claim: str = typer.Argument(help="The claim to analyze"),
    context: str | None = typer.Option(None, "--context", "-c", help="Background context"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output raw JSON"),
) -> None:
    """Test a hypothesis through structured analysis (steelman → redteam → judge)."""
    try:
        llm = _get_llm()
        # Retrieve prior context if available
        prior_hypotheses = None
        existing_hypotheses = None
        try:
            store = _get_store()
            prior_hypotheses, existing_hypotheses = _retrieve_context(claim, store)
        except Exception:
            store = None  # Will create fresh for persistence

        verdict = asyncio.run(
            analyze(
                claim=claim, llm=llm, context=context,
                prior_hypotheses=prior_hypotheses,
                existing_hypotheses=existing_hypotheses,
            )
        )
    except Exception as e:
        console.print(_format_error(e))
        raise typer.Exit(code=1) from e

    if json_output:
        typer.echo(json.dumps(verdict.model_dump(), indent=2))
        return

    # Rich formatted output
    _render_verdict(claim, verdict)

    # Persist to DB
    try:
        if store is None:
            store = _get_store()
        h = store.save_hypothesis(
            claim=claim,
            source="manual",
            context=context,
            confidence=verdict.confidence,
            tags=verdict.tags,
            source_ref=json.dumps(verdict.model_dump()),
        )
        # Store embedding for future retrieval
        try:
            from forge.retrieve.embeddings import get_embedder

            embedder = get_embedder()
            vec = embedder.embed(claim)
            store.update_hypothesis(h.id, embedding=vec.tobytes())
        except Exception:
            pass  # Embedding is optional
        # Save relations extracted by the judge
        try:
            from forge.analyze.relations import save_verdict_relations

            save_verdict_relations(h.id, verdict, store)
        except Exception:
            pass  # Relations are optional
        # Flag high-confidence claims for swarm simulation
        if verdict.confidence > 60:
            try:
                store.save_simulation(
                    mode="claim_test",
                    seed_text=claim,
                    seed_context=context,
                )
                console.print(
                    f"\n[dim]This claim scored {verdict.confidence} confidence. "
                    f"Run `forge simulate '{claim}'` for deeper analysis.[/dim]"
                )
            except Exception:
                pass  # Queuing is optional
    except Exception as e:
        console.print(f"[yellow]Warning: could not persist result: {e}[/yellow]")


def _render_verdict(claim: str, verdict) -> None:
    """Render a verdict as a Rich panel."""
    from forge.analyze.structured import Verdict

    assert isinstance(verdict, Verdict)

    color = "green" if verdict.confidence >= 60 else "yellow" if verdict.confidence >= 40 else "red"

    panel_content = (
        f"[bold]Position:[/bold] {verdict.position}\n"
        f"[bold]Confidence:[/bold] [{color}]{verdict.confidence}/100[/{color}]\n\n"
        f"[bold]Best case FOR:[/bold] {verdict.steelman_arg}\n"
        f"[bold]Best case AGAINST:[/bold] {verdict.redteam_arg}\n\n"
        f"[bold]Synthesis:[/bold] {verdict.synthesis}\n"
    )

    if verdict.conditions:
        panel_content += "\n[bold]Conditions:[/bold]\n"
        for cond in verdict.conditions:
            panel_content += f"  • {cond}\n"

    if verdict.tags:
        panel_content += f"\n[bold]Tags:[/bold] {', '.join(verdict.tags)}"

    console.print(Panel(panel_content, title="[bold]FORGE Analysis[/bold]", subtitle=claim))


def _retrieve_context(claim: str, store: Store) -> tuple[str | None, str | None]:
    """Retrieve prior context for a claim if embeddings are available."""
    from forge.retrieve.context import retrieve_prior_context
    from forge.retrieve.embeddings import get_embedder

    embedder = get_embedder()
    claim_vec = embedder.embed(claim)
    prior_text, existing_text = retrieve_prior_context(claim_vec, store, limit=3)
    return prior_text or None, existing_text or None


@app.command()
def history(
    status: str | None = typer.Option(None, "--status", "-s", help="Filter by status"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
) -> None:
    """Show past hypotheses."""
    from rich.table import Table

    store = _get_store()
    hypotheses = store.list_hypotheses(status=status)[:limit]

    if not hypotheses:
        console.print("No hypotheses yet.")
        return

    table = Table(title="Hypotheses")
    table.add_column("ID", style="dim", max_width=12)
    table.add_column("Claim", max_width=60)
    table.add_column("Conf", justify="right")
    table.add_column("Status")
    table.add_column("Created", style="dim")

    for h in hypotheses:
        claim_short = h.claim[:57] + "..." if len(h.claim) > 60 else h.claim
        conf_color = "green" if h.confidence >= 60 else "yellow" if h.confidence >= 40 else "red"
        table.add_row(
            h.id[:12],
            claim_short,
            f"[{conf_color}]{h.confidence}[/{conf_color}]",
            h.status,
            h.created_at[:10],
        )

    console.print(table)


def _check_llm_health() -> tuple[str, str | None]:
    """Check llama.cpp health. Returns (status, model_name)."""
    settings = Settings()
    try:
        resp = httpx.get(f"{settings.llama_url}/v1/models", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data", [])
        model_name = models[0]["id"] if models else "unknown"
        return ("healthy", model_name)
    except Exception:
        return ("unreachable", None)


@app.command()
def status() -> None:
    """Show system health and DB stats."""
    from rich.table import Table

    store = _get_store()

    # DB stats
    h_counts = store.count_hypotheses_by_status()
    total_h = sum(h_counts.values())
    e_count = store.count_evidence()
    f_count = store.count_feedback()

    console.print("[bold]Database[/bold]")
    table = Table(show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Hypotheses", str(total_h))
    for s, c in sorted(h_counts.items()):
        table.add_row(f"  {s}", str(c))
    table.add_row("Evidence", str(e_count))
    table.add_row("Feedback", str(f_count))
    console.print(table)

    # LLM health
    console.print("\n[bold]LLM Backend[/bold]")
    llm_status, model_name = _check_llm_health()
    if llm_status == "healthy":
        console.print("  Status: [green]healthy[/green]")
        console.print(f"  Model: {model_name}")
    else:
        console.print("  Status: [red]unreachable[/red]")


@app.command()
def graph(
    hypothesis_id: str = typer.Argument(help="Hypothesis ID to show graph for"),
) -> None:
    """Show a hypothesis and its relations as a tree."""
    from rich.tree import Tree

    store = _get_store()
    h = store.get_hypothesis(hypothesis_id)

    if h is None:
        console.print(f"Hypothesis [bold]{hypothesis_id}[/bold] not found.")
        return

    tree = Tree(f"[bold]{h.claim}[/bold] (conf: {h.confidence}, {h.status})")

    relations = store.list_relations_for_hypothesis(h.id)
    if not relations:
        tree.add("[dim]No relations[/dim]")
    else:
        for rel in relations:
            # Determine direction
            if rel.source_id == h.id:
                other_id = rel.target_id
                direction = "→"
            else:
                other_id = rel.source_id
                direction = "←"
            other = store.get_hypothesis(other_id)
            other_label = other.claim if other else other_id
            tree.add(
                f"{direction} [cyan]{rel.relation_type}[/cyan]: "
                f"{other_label}"
                + (f" [dim]({rel.reasoning})[/dim]" if rel.reasoning else "")
            )

    console.print(tree)


@app.command()
def turns(
    simulation_id: str | None = typer.Argument(None, help="Simulation ID to inspect"),
    round_num: int | None = typer.Option(None, "--round", "-r", help="Filter by round"),
    agent: str | None = typer.Option(None, "--agent", "-a", help="Filter by archetype"),
    detail: bool = typer.Option(False, "--detail", "-d", help="Show full LLM content"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output raw JSON"),
    md: bool = typer.Option(False, "--md", "--markdown", help="Output as markdown (LLM-friendly)"),
    pager: bool = typer.Option(False, "--pager", help="Use interactive pager"),
) -> None:
    """Browse LLM responses from simulation runs."""
    from rich.table import Table

    store = _get_store()

    if simulation_id is None:
        # List recent simulations
        sims = store.list_simulations()
        if not sims:
            console.print("No simulations yet.")
            return

        table = Table(title="Simulations")
        table.add_column("ID", style="dim", no_wrap=True)
        table.add_column("Scenario", max_width=50)
        table.add_column("Agents", justify="right")
        table.add_column("Rounds", justify="right")
        table.add_column("Status")
        table.add_column("Date", style="dim")

        for s in sims:
            table.add_row(
                s.id,
                s.seed_text[:47] + "..." if len(s.seed_text) > 50 else s.seed_text,
                str(s.agent_count or "-"),
                str(s.rounds or "-"),
                s.status,
                s.started_at[:10] if s.started_at else "-",
            )
        console.print(table)
        return

    # Verify simulation exists (support prefix matching)
    sim = store.get_simulation(simulation_id) or store.find_simulation_by_prefix(simulation_id)
    if sim is None:
        console.print(f"[red]Simulation {simulation_id} not found.[/red]")
        raise typer.Exit(code=1)

    rows = store.list_turns_with_agent(
        sim.id, round=round_num, archetype=agent,
    )

    if not rows:
        console.print("No turns found.")
        return

    if json_output:
        import json as json_mod
        typer.echo(json_mod.dumps(rows, indent=2, default=str))
        return

    if md:
        from forge.cli_markdown import render_turns_markdown
        predictions = store.list_predictions(simulation_id=sim.id)
        typer.echo(render_turns_markdown(rows, sim, predictions=predictions))
        return

    output_console = Console() if pager else console

    if detail:
        _render_turns_detail(rows, output_console, pager)
    else:
        _render_turns_table(rows, output_console, pager)


def _safe_parse(content: str | None) -> dict:
    """Safely parse JSON content, returning empty dict on failure."""
    if not content:
        return {}
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {}


def _render_turns_table(rows: list[dict], target: Console, use_pager: bool) -> None:
    """Render turns as a summary table."""
    from rich.table import Table

    table = Table(title="Simulation Turns")
    table.add_column("Round", justify="right")
    table.add_column("Agent")
    table.add_column("Position")
    table.add_column("Conf", justify="right")
    table.add_column("Type")
    table.add_column("Reasoning", max_width=50)

    for row in rows:
        content = _safe_parse(row.get("content"))
        reasoning = content.get("reasoning", content.get("key_insight", ""))
        if len(reasoning) > 50:
            reasoning = reasoning[:47] + "..."

        conf = row.get("confidence")
        conf_str = str(conf) if conf is not None else "-"

        table.add_row(
            str(row["round"]),
            row["archetype"],
            row.get("position") or "-",
            conf_str,
            row["turn_type"],
            reasoning,
        )

    if use_pager:
        with target.pager():
            target.print(table)
    else:
        target.print(table)


def _render_turns_detail(rows: list[dict], target: Console, use_pager: bool) -> None:
    """Render turns with full content in panels."""
    panels = []
    for row in rows:
        content = row["content"]
        try:
            parsed = json.loads(content)
            formatted = json.dumps(parsed, indent=2)
        except (json.JSONDecodeError, TypeError):
            formatted = content or "(empty)"

        title = (
            f"[bold]Round {row['round']}[/bold] | "
            f"{row['archetype']} | "
            f"{row.get('position', 'neutral')} ({row.get('confidence', '?')}%)"
        )
        panels.append(Panel(formatted, title=title))

    if use_pager:
        with target.pager():
            for p in panels:
                target.print(p)
    else:
        for p in panels:
            target.print(p)


@app.command()
def simulate(
    scenario: str = typer.Argument(help="The scenario to simulate"),
    context: str | None = typer.Option(None, "--context", "-c", help="Background context"),
    agents: int | None = typer.Option(None, "--agents", "-a", help="Number of agents"),
    rounds: int | None = typer.Option(None, "--rounds", "-r", help="Number of rounds"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output raw JSON"),
    md: bool = typer.Option(False, "--md", "--markdown", help="Output as markdown (LLM-friendly)"),
) -> None:
    """Run a full swarm simulation on a scenario."""
    from forge.swarm.arena import run_simulation
    from forge.swarm.population import generate_population
    from forge.swarm.predictions import extract_predictions

    settings = Settings()
    agent_count = agents or settings.default_swarm_size
    round_count = rounds or settings.simulation_rounds

    try:
        llm = _get_llm()
        store = _get_store()
        seed = SeedMaterial(text=scenario, context=context)

        # Run the full pipeline
        result = asyncio.run(_run_simulate(
            seed, llm, store, agent_count, round_count,
            generate_population, run_simulation, extract_predictions,
        ))
    except Exception as e:
        console.print(_format_error(e))
        raise typer.Exit(code=1) from e

    if json_output:
        output = {
            "simulation_id": result["simulation"].id,
            "status": result["simulation"].status,
            "agent_count": result["simulation"].agent_count,
            "rounds": result["simulation"].rounds,
            "duration_seconds": result["sim_result"].duration_seconds,
            "majority_position": result["consensus"].majority_position,
            "majority_confidence": result["consensus"].majority_confidence,
            "majority_fraction": result["consensus"].majority_fraction,
            "predictions": [
                {"id": p.id, "claim": p.claim, "confidence": p.confidence}
                for p in result["predictions"]
            ],
        }
        typer.echo(json.dumps(output, indent=2))
        return

    if md:
        from forge.cli_markdown import render_turns_markdown
        sim_id = result["simulation"].id
        rows = store.list_turns_with_agent(sim_id)
        predictions = store.list_predictions(simulation_id=sim_id)
        typer.echo(render_turns_markdown(
            rows, result["simulation"], predictions=predictions,
        ))
        return

    _render_simulation(result)


async def _run_simulate(seed, llm, store, agent_count, round_count,
                        gen_pop, run_sim, extract_preds):
    """Orchestrate the full simulation pipeline."""
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

    from forge.cli_render import summarize_round

    population = await gen_pop(seed, llm, store, count=agent_count)
    if not population:
        raise RuntimeError(
            f"No agents generated — LLM failed to produce valid personas "
            f"(requested {agent_count}). Check LLM server output."
        )

    total_work = agent_count * round_count
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    )
    task_id = progress.add_task("Simulating...", total=total_work)
    round_summaries: list[str] = []

    def on_turn(turn, round_num, agent):
        """Advance progress bar for each completed agent."""
        progress.update(task_id, advance=1, description=f"Round {round_num}")

    def on_round_complete(round_num, turns, pop, prev_turns=None):
        """Collect round summary for display after progress completes."""
        summary = summarize_round(
            turns,
            round_num=round_num,
            expected_agents=len(pop),
            prev_turns=prev_turns,
        )
        round_summaries.append(summary.format_line())

    console.print(
        f"[dim]Running {round_count} rounds with "
        f"{agent_count} agents...[/dim]\n"
    )
    with progress:
        sim_result = await run_sim(
            seed, population, llm, store,
            rounds=round_count, max_concurrent=2,
            on_turn=on_turn, on_round_complete=on_round_complete,
        )

    # Print round summaries after progress bar finishes
    for summary_line in round_summaries:
        console.print(summary_line)
    console.print()  # blank line before panel

    predictions = await extract_preds(
        seed, sim_result.consensus, llm, store, sim_result.simulation.id,
        agent_count=sim_result.simulation.agent_count,
    )
    return {
        "simulation": sim_result.simulation,
        "sim_result": sim_result,
        "consensus": sim_result.consensus,
        "predictions": predictions,
    }


def _render_simulation(result: dict) -> None:
    """Render simulation results as Rich output."""
    from rich.table import Table

    sim = result["simulation"]
    consensus = result["consensus"]
    predictions = result["predictions"]
    sim_result = result["sim_result"]

    # Header panel
    color = (
        "green" if consensus.majority_confidence >= 60
        else "yellow" if consensus.majority_confidence >= 40
        else "red"
    )
    duration = sim_result.duration_seconds
    duration_color = (
        "red" if duration >= 1200
        else "yellow" if duration >= 600
        else "dim"
    )
    panel_content = (
        f"[bold]Agents:[/bold] {sim.agent_count}  "
        f"[bold]Rounds:[/bold] {sim.rounds}  "
        f"[bold]Duration:[/bold] [{duration_color}]{duration:.1f}s[/{duration_color}]\n\n"
        f"[bold]Majority:[/bold] {consensus.majority_position} "
        f"[{color}]({consensus.majority_confidence:.0f}% confidence, "
        f"{consensus.majority_fraction:.0%} of agents)[/{color}]"
    )

    # Confidence trend across rounds
    if consensus.confidence_trend:
        trend_parts = [f"{c:.0f}%" for c in consensus.confidence_trend]
        panel_content += f"\n[bold]Confidence:[/bold] {' → '.join(trend_parts)}"

    # Failed agents (compare expected vs actual round 3 participants)
    r3_count = len([t for t in sim_result.turns if t.round == 3])
    if r3_count < sim.agent_count:
        failed = sim.agent_count - r3_count
        panel_content += (
            f"\n[yellow]({failed} agent(s) did not complete round 3)[/yellow]"
        )

    if consensus.dissent_clusters:
        panel_content += "\n\n[bold]Dissent:[/bold]"
        for cluster in consensus.dissent_clusters:
            panel_content += (
                f"\n  {cluster.position}: {cluster.agent_count} agents "
                f"({cluster.avg_confidence:.0f}% confidence)"
            )

    if consensus.conviction_shifts:
        panel_content += "\n\n[bold]Changed minds:[/bold]"
        for shift in consensus.conviction_shifts:
            panel_content += (
                f"\n  {shift.archetype}: {shift.from_position} → {shift.to_position}"
            )

    # Diagnostics summary
    if sim_result.diagnostics.total_failures > 0:
        diag = sim_result.diagnostics.format_summary()
        panel_content += (
            f"\n\n[yellow bold]Errors ({sim_result.diagnostics.total_failures}):[/yellow bold]"
            f"\n{diag}"
        )

    console.print(Panel(panel_content, title="[bold]FORGE Simulation[/bold]"))

    # Predictions table
    if predictions:
        table = Table(title="Predictions")
        table.add_column("Claim", max_width=60)
        table.add_column("Conf", justify="right")
        table.add_column("Consensus", justify="right")
        table.add_column("Deadline", style="dim")

        for p in predictions:
            conf_color = (
                "green" if p.confidence >= 60
                else "yellow" if p.confidence >= 40
                else "red"
            )
            table.add_row(
                p.claim[:57] + "..." if len(p.claim) > 60 else p.claim,
                f"[{conf_color}]{p.confidence}[/{conf_color}]",
                f"{p.consensus_strength:.0%}" if p.consensus_strength else "-",
                p.resolution_deadline[:10] if p.resolution_deadline else "-",
            )
        console.print(table)
    else:
        console.print("[dim]No predictions extracted.[/dim]")


# ------------------------------------------------------------------
# Feed management commands
# ------------------------------------------------------------------


@feed_app.command("add")
def feed_add(
    url: str = typer.Argument(help="RSS feed URL"),
    name: str | None = typer.Option(None, "--name", "-n", help="Feed name"),
    interval: int = typer.Option(240, "--interval", "-i", help="Poll interval (minutes)"),
) -> None:
    """Add an RSS feed to watch."""
    store = _get_store()
    feed_name = name or url.split("/")[2]  # domain as default name
    feed = store.save_feed(name=feed_name, url=url, poll_interval_minutes=interval)
    console.print(f"Added feed [bold]{feed.name}[/bold] ({feed.id})")


@feed_app.command("list")
def feed_list(
    active_only: bool = typer.Option(False, "--active", "-a", help="Show only active feeds"),
) -> None:
    """List all RSS feeds."""
    from rich.table import Table

    store = _get_store()
    feeds = store.list_feeds(active=True if active_only else None)

    if not feeds:
        console.print("No feeds configured.")
        return

    table = Table(title="Feeds")
    table.add_column("ID", style="dim", max_width=12)
    table.add_column("Name")
    table.add_column("URL", max_width=50)
    table.add_column("Active")
    table.add_column("Last Polled", style="dim")

    for f in feeds:
        table.add_row(
            f.id[:12],
            f.name,
            f.url[:47] + "..." if len(f.url) > 50 else f.url,
            "[green]yes[/green]" if f.active else "[red]no[/red]",
            f.last_polled_at[:16] if f.last_polled_at else "never",
        )
    console.print(table)


@feed_app.command("remove")
def feed_remove(
    feed_id: str = typer.Argument(help="Feed ID to deactivate"),
) -> None:
    """Deactivate an RSS feed."""
    store = _get_store()
    result = store.update_feed(feed_id, active=0)
    if result is None:
        console.print(f"Feed [bold]{feed_id}[/bold] not found.")
        raise typer.Exit(code=1)
    console.print(f"Deactivated feed [bold]{result.name}[/bold]")


@feed_app.command("poll")
def feed_poll() -> None:
    """Poll all active feeds for new articles."""
    from forge.ingest.rss import poll_all_feeds

    store = _get_store()
    results = poll_all_feeds(store)

    if not results:
        console.print("No feeds due for polling.")
        return

    total = sum(results.values())
    console.print(f"Polled {len(results)} feed(s), found {total} new article(s).")
    for url, count in results.items():
        console.print(f"  {url}: {count} new")


# ------------------------------------------------------------------
# Resolution + predictions commands
# ------------------------------------------------------------------


@app.command()
def resolve(
    prediction_id: str = typer.Argument(help="Prediction ID (p_...)"),
    true: bool = typer.Option(False, "--true", help="Resolve as true"),
    false_: bool = typer.Option(False, "--false", help="Resolve as false"),
    partial: bool = typer.Option(False, "--partial", help="Resolve as partial"),
    note: str | None = typer.Option(None, "--note", "-n", help="Resolution note"),
) -> None:
    """Resolve a prediction as true, false, or partial."""
    from forge.calibrate.resolver import resolve_prediction

    outcome_map = {"true": true, "false": false_, "partial": partial}
    selected = [k for k, v in outcome_map.items() if v]

    if len(selected) != 1:
        console.print("[red]Specify exactly one of --true, --false, or --partial[/red]")
        raise typer.Exit(code=1)

    store = _get_store()
    try:
        resolve_prediction(store, prediction_id, selected[0], note=note)
        console.print(
            f"Resolved [bold]{prediction_id}[/bold] as "
            f"[{'green' if selected[0] == 'true' else 'red'}]{selected[0]}[/]"
        )
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from e


@app.command()
def predictions(
    pending: bool = typer.Option(False, "--pending", help="Show pending only"),
    resolved: bool = typer.Option(False, "--resolved", help="Show resolved only"),
    overdue: bool = typer.Option(False, "--overdue", help="Show overdue only"),
) -> None:
    """List predictions with their resolution status."""
    from rich.table import Table

    store = _get_store()

    if overdue:
        preds = store.list_predictions_past_deadline()
    elif resolved:
        preds = store.list_resolved_predictions()
    elif pending:
        preds = store.list_predictions_pending()
    else:
        preds = store.list_predictions()

    if not preds:
        console.print("No predictions found.")
        return

    table = Table(title="Predictions")
    table.add_column("ID", style="dim", max_width=12)
    table.add_column("Claim", max_width=50)
    table.add_column("Conf", justify="right")
    table.add_column("Status")
    table.add_column("Deadline", style="dim")

    for p in preds:
        status = p.resolved_as or "pending"
        color = {"true": "green", "false": "red", "partial": "yellow"}.get(status, "dim")
        table.add_row(
            p.id[:12],
            p.claim[:47] + "..." if len(p.claim) > 50 else p.claim,
            str(p.confidence),
            f"[{color}]{status}[/{color}]",
            p.resolution_deadline[:10] if p.resolution_deadline else "-",
        )
    console.print(table)


# ------------------------------------------------------------------
# Calibration command
# ------------------------------------------------------------------


@app.command()
def calibration() -> None:
    """Show calibration report: accuracy, Brier score, bucket breakdown."""
    from rich.table import Table

    from forge.calibrate.scorer import compute_calibration

    store = _get_store()
    resolved = store.list_resolved_predictions()

    if not resolved:
        console.print("No resolved predictions yet. Use `forge resolve` to resolve predictions.")
        return

    report = compute_calibration(resolved)

    console.print("\n[bold]Calibration Report[/bold]")
    console.print(f"  Total predictions: {report['total']}")
    console.print(f"  Resolved: {report['resolved']}")
    console.print(f"  Accuracy: {report['accuracy']:.1%}")
    console.print(f"  Brier score: {report['brier_score']:.4f}")

    if report["buckets"]:
        table = Table(title="Calibration Buckets")
        table.add_column("Confidence", justify="center")
        table.add_column("Total", justify="right")
        table.add_column("Correct", justify="right")
        table.add_column("Accuracy", justify="right")

        for b in report["buckets"]:
            table.add_row(
                b["bucket"],
                str(b["total"]),
                str(b["correct"]),
                f"{b['accuracy']:.0%}",
            )
        console.print(table)


# ------------------------------------------------------------------
# Pipeline commands
# ------------------------------------------------------------------


@app.command()
def run(
    once: bool = typer.Option(False, "--once", help="Run a single pipeline cycle"),
    auto_simulate: bool = typer.Option(
        False, "--auto-simulate", help="Auto-simulate top claims after ingestion",
    ),
) -> None:
    """Run the ingestion pipeline (continuous or single cycle)."""
    from forge.pipeline.runner import run_pipeline_once
    from forge.pipeline.scheduler import run_scheduled

    settings = Settings()
    store = _get_store()
    llm = _get_llm()

    sim_enabled = auto_simulate or settings.auto_simulate
    sim_kwargs = {
        "auto_simulate": sim_enabled,
        "auto_simulate_top_n": settings.auto_simulate_top_n,
        "auto_simulate_min_confidence": settings.auto_simulate_min_confidence,
        "auto_simulate_agent_count": settings.auto_simulate_agent_count,
        "auto_simulate_rounds": settings.auto_simulate_rounds,
    }

    if once:
        result = asyncio.run(run_pipeline_once(store, llm, **sim_kwargs))
        ingestion = result["ingestion"]
        parts = [
            f"Fetched {ingestion['articles_fetched']} article(s)",
            f"extracted {ingestion['claims_extracted']} claim(s)",
            f"{result['overdue_predictions']} overdue prediction(s)",
        ]
        if sim_enabled:
            parts.append(f"{result['auto_simulations']} auto-simulation(s)")
        console.print(", ".join(parts) + ".")
    else:
        console.print(
            f"Starting continuous pipeline (interval: {settings.poll_interval_minutes}m). "
            "Press Ctrl+C to stop."
        )
        try:
            asyncio.run(run_scheduled(
                store, llm, settings.poll_interval_minutes, **sim_kwargs,
            ))
        except KeyboardInterrupt:
            console.print("\nPipeline stopped.")


# ------------------------------------------------------------------
# Phase 4: Feedback commands
# ------------------------------------------------------------------


@app.command()
def endorse(
    hypothesis_id: str = typer.Argument(help="Hypothesis ID (h_...)"),
    note: str | None = typer.Option(None, "--note", "-n", help="Optional note"),
) -> None:
    """Endorse a hypothesis (increases human_endorsed count)."""
    store = _get_store()
    h = store.get_hypothesis(hypothesis_id)
    if h is None:
        console.print(f"[red]Hypothesis {hypothesis_id} not found.[/red]")
        raise typer.Exit(code=1)

    store.update_hypothesis(h.id, human_endorsed=h.human_endorsed + 1)
    store.save_feedback("endorse", hypothesis_id=h.id, note=note)
    console.print(
        f"Endorsed [bold]{h.id}[/bold] "
        f"(endorsed: {h.human_endorsed + 1})"
    )


@app.command()
def reject(
    hypothesis_id: str = typer.Argument(help="Hypothesis ID (h_...)"),
    note: str | None = typer.Option(None, "--note", "-n", help="Optional note"),
) -> None:
    """Reject a hypothesis (increases human_rejected count)."""
    store = _get_store()
    h = store.get_hypothesis(hypothesis_id)
    if h is None:
        console.print(f"[red]Hypothesis {hypothesis_id} not found.[/red]")
        raise typer.Exit(code=1)

    store.update_hypothesis(h.id, human_rejected=h.human_rejected + 1)
    store.save_feedback("reject", hypothesis_id=h.id, note=note)
    console.print(
        f"Rejected [bold]{h.id}[/bold] "
        f"(rejected: {h.human_rejected + 1})"
    )


# ------------------------------------------------------------------
# Phase 4: Digest, leaderboard, evolve, export
# ------------------------------------------------------------------


@app.command()
def brief() -> None:
    """Generate and display today's daily digest."""
    from forge.export.digest import generate_digest

    store = _get_store()
    digest = generate_digest(store)
    console.print(digest.to_markdown())


@app.command()
def leaderboard(
    limit: int = typer.Option(10, "--limit", "-n", help="Max agents to show"),
) -> None:
    """Show top-performing agent archetypes."""
    from rich.table import Table

    from forge.evolve.agent_evolution import get_leaderboard

    store = _get_store()
    leaders = get_leaderboard(store, limit=limit)

    if not leaders:
        console.print("No scored agents yet.")
        return

    table = Table(title="Agent Leaderboard")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Archetype")
    table.add_column("Calibration", justify="right")
    table.add_column("Sims", justify="right")
    table.add_column("Correct", justify="right")
    table.add_column("Incorrect", justify="right")

    for i, a in enumerate(leaders, 1):
        score = (
            f"{a.calibration_score:.0%}"
            if a.calibration_score is not None
            else "N/A"
        )
        table.add_row(
            str(i),
            a.archetype,
            score,
            str(a.simulations_participated),
            str(a.predictions_correct),
            str(a.predictions_incorrect),
        )
    console.print(table)


@app.command()
def evolve() -> None:
    """Run hypothesis and agent evolution cycles."""
    from forge.evolve.agent_evolution import run_agent_evolution
    from forge.evolve.selection import run_evolution_cycle

    store = _get_store()

    h_result = run_evolution_cycle(store)
    a_result = run_agent_evolution(store)

    console.print("[bold]Evolution Cycle Complete[/bold]")
    console.print(f"  Hypotheses culled: {len(h_result.culled)}")
    console.print(f"  Hypotheses dormant: {len(h_result.dormant)}")
    console.print(f"  Hypotheses promoted: {len(h_result.promoted)}")
    console.print(f"  Confidence boosted: {len(h_result.boosted)}")
    console.print(f"  Confirmed: {len(h_result.confirmed)}")
    console.print(f"  Forked: {len(h_result.forked)}")
    console.print(f"  Agents deactivated: {len(a_result.deactivated)}")


@app.command(name="export")
def export_vault(
    vault_path: str = typer.Argument(
        help="Path to Obsidian vault directory",
    ),
) -> None:
    """Export knowledge graph to Obsidian vault (one-way sync)."""
    from forge.export.obsidian import render_vault

    store = _get_store()
    render_vault(store, vault_path)
    console.print(f"Exported to [bold]{vault_path}[/bold]")
