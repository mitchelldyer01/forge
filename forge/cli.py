"""CLI entrypoint for FORGE — Typer app."""

from __future__ import annotations

import asyncio
import json

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
