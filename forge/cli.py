"""CLI entrypoint for FORGE — Typer application."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from forge.analyze.structured import run_structured_analysis
from forge.config import get_settings
from forge.db.store import Store
from forge.llm.client import LLMClient

app = typer.Typer(name="forge", help="FORGE — Calibrated Prediction Engine")
console = Console()


@app.command()
def test(
    claim: str = typer.Argument(..., help="The claim to analyze"),
    context: str | None = typer.Option(None, "--context", "-c", help="Background context"),
    output_json: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """Run structured analysis (steelman -> redteam -> judge) on a claim."""
    settings = get_settings()
    llm = LLMClient(base_url=settings.llama_url, timeout=settings.llama_timeout)

    try:
        verdict = asyncio.run(
            run_structured_analysis(llm, claim=claim, context=context)
        )
    except Exception as exc:
        console.print(f"[red]Pipeline failed:[/red] {exc}")
        raise typer.Exit(code=1) from None

    if output_json:
        typer.echo(verdict.model_dump_json(indent=2))
        return

    # Rich formatted output
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("Position", verdict.position)
    table.add_row("Confidence", f"{verdict.confidence}/100")
    table.add_row("Synthesis", verdict.synthesis)
    if verdict.conditions:
        table.add_row("Conditions", ", ".join(verdict.conditions))
    if verdict.tags:
        table.add_row("Tags", ", ".join(verdict.tags))

    console.print(Panel(table, title="[bold]FORGE Verdict[/bold]", subtitle=claim[:60]))


@app.command()
def history(
    status_filter: str | None = typer.Option(None, "--status", "-s", help="Filter by status"),
    limit: int | None = typer.Option(None, "--limit", "-n", help="Max results"),
) -> None:
    """Show a table of past hypotheses."""
    settings = get_settings()
    store = Store(settings.db_path)

    hypotheses = store.list_hypotheses(status=status_filter)
    if limit is not None:
        hypotheses = hypotheses[:limit]

    if not hypotheses:
        console.print("[dim]No hypotheses yet.[/dim]")
        return

    table = Table(title="Hypotheses")
    table.add_column("ID", style="dim", max_width=12)
    table.add_column("Claim", max_width=50)
    table.add_column("Confidence", justify="right")
    table.add_column("Status")
    table.add_column("Created", style="dim")

    for h in hypotheses:
        table.add_row(
            h.id[:12],
            h.claim[:50],
            str(h.confidence),
            h.status,
            h.created_at[:19],
        )

    console.print(table)


@app.command()
def status() -> None:
    """Show system health and database statistics."""
    settings = get_settings()

    try:
        store = Store(settings.db_path)
        hypotheses = store.list_hypotheses()
        alive = sum(1 for h in hypotheses if h.status == "alive")
        dead = sum(1 for h in hypotheses if h.status == "dead")

        table = Table(title="FORGE Status", show_header=False)
        table.add_column(style="bold")
        table.add_column()
        table.add_row("DB Path", settings.db_path)
        table.add_row("Hypotheses (alive)", str(alive))
        table.add_row("Hypotheses (dead)", str(dead))
        table.add_row("Hypotheses (total)", str(len(hypotheses)))
        table.add_row("LLM URL", settings.llama_url)

        console.print(table)
    except Exception as exc:
        console.print(f"[red]Status check failed:[/red] {exc}")
        raise typer.Exit(code=1) from None
