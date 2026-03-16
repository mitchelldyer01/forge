"""Prompt loader — reads markdown templates and renders with Jinja2."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Template

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str, **variables: str) -> str:
    """Load a prompt template by name and render with variables.

    Args:
        name: Prompt name (e.g. "steelman", "redteam", "judge").
        **variables: Template variables (claim, context, steelman_output, etc.)

    Returns:
        Rendered prompt string.

    Raises:
        FileNotFoundError: If prompt file doesn't exist.
    """
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    template = Template(path.read_text())
    return template.render(**variables)
