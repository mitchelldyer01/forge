"""CLI entrypoint for FORGE — Typer app."""

from __future__ import annotations

import asyncio
import json

import httpx
import typer
from rich.console import Console
from rich.panel import Panel

from forge.analyze.structured import analyze
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


@app.callback()
def main() -> None:
    """FORGE — Calibrated Prediction Engine."""


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
        console.print(f"[red]Error:[/red] {e}", style="bold red")
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
            from forge.retrieve.embeddings import Embedder

            embedder = Embedder()
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
    from forge.retrieve.embeddings import Embedder

    embedder = Embedder()
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
def simulate(
    scenario: str = typer.Argument(help="The scenario to simulate"),
    context: str | None = typer.Option(None, "--context", "-c", help="Background context"),
    agents: int | None = typer.Option(None, "--agents", "-a", help="Number of agents"),
    rounds: int | None = typer.Option(None, "--rounds", "-r", help="Number of rounds"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output raw JSON"),
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
        console.print(f"[red]Error:[/red] {e}", style="bold red")
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

    _render_simulation(result)


async def _run_simulate(seed, llm, store, agent_count, round_count,
                        gen_pop, run_sim, extract_preds):
    """Orchestrate the full simulation pipeline."""
    population = await gen_pop(seed, llm, store, count=agent_count)
    sim_result = await run_sim(
        seed, population, llm, store, rounds=round_count, max_concurrent=8,
    )
    predictions = await extract_preds(
        seed, sim_result.consensus, llm, store, sim_result.simulation.id,
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
    panel_content = (
        f"[bold]Agents:[/bold] {sim.agent_count}  "
        f"[bold]Rounds:[/bold] {sim.rounds}  "
        f"[bold]Duration:[/bold] {sim_result.duration_seconds:.1f}s\n\n"
        f"[bold]Majority:[/bold] {consensus.majority_position} "
        f"[{color}]({consensus.majority_confidence:.0f}% confidence, "
        f"{consensus.majority_fraction:.0%} of agents)[/{color}]"
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
