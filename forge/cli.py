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

app = typer.Typer(
    name="forge",
    help="FORGE — Calibrated Prediction Engine",
    invoke_without_command=True,
)
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
        verdict = asyncio.run(
            analyze(claim=claim, llm=llm, context=context)
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
        store = _get_store()
        store.save_hypothesis(
            claim=claim,
            source="manual",
            context=context,
            confidence=verdict.confidence,
            tags=verdict.tags,
            source_ref=json.dumps(verdict.model_dump()),
        )
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
