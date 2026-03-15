"""Prompt loading and rendering for structured analysis."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str, **variables: str | None) -> str:
    """Load and render a prompt template by name.

    Args:
        name: Prompt name (e.g. "steelman", "redteam", "judge")
        **variables: Template variables (claim, context, steelman_arg, etc.)

    Returns:
        Rendered prompt string.

    Raises:
        FileNotFoundError: If the prompt file doesn't exist.
    """
    template_path = PROMPTS_DIR / f"{name}.md"
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")

    env = Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        keep_trailing_newline=True,
    )
    template = env.get_template(f"{name}.md")
    return template.render(**{k: v or "" for k, v in variables.items()})
