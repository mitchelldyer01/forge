"""Prompt loader for swarm simulation prompts."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Template

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_swarm_prompt(name: str, **variables: str) -> str:
    """Load and render a swarm prompt template by name.

    Args:
        name: Prompt name (e.g. "persona_generator", "reaction").
        **variables: Template variables.

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
