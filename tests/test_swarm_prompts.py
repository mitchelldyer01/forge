"""
Tests for swarm prompt templates — reaction.md, persona_generator.md, convergence.md.

Mirrors: forge/swarm/prompts/
"""

import pytest

from forge.swarm.prompts import load_swarm_prompt


@pytest.mark.unit
class TestReactionPromptConfidenceAnchor:
    def test_reaction_prompt_contains_confidence_anchor(self):
        """Reaction prompt includes the confidence_anchor from persona."""
        result = load_swarm_prompt(
            "reaction",
            agent_name="Test Agent",
            agent_background="Test background",
            agent_archetype="analyst",
            agent_expertise="economics",
            risk_appetite="medium",
            optimism_bias="realist",
            reasoning_style="analytical",
            seed_text="Test scenario",
            seed_context="",
            confidence_anchor="low (20-45)",
        )
        assert "low (20-45)" in result

    def test_reaction_prompt_contains_no_default_75(self):
        """Reaction prompt explicitly tells agents not to default to 75."""
        result = load_swarm_prompt(
            "reaction",
            agent_name="Test Agent",
            agent_background="Test background",
            agent_archetype="analyst",
            agent_expertise="economics",
            risk_appetite="medium",
            optimism_bias="realist",
            reasoning_style="analytical",
            seed_text="Test scenario",
            seed_context="",
            confidence_anchor="medium (46-70)",
        )
        assert "75" in result or "default" in result.lower()


@pytest.mark.unit
class TestPersonaGeneratorPrompt:
    def test_persona_generator_prompt_contains_confidence_anchor_field(self):
        """Persona generator prompt mentions confidence_anchor in schema."""
        result = load_swarm_prompt(
            "persona_generator",
            seed_text="Test scenario",
            seed_context="",
            count="5",
        )
        assert "confidence_anchor" in result

    def test_persona_generator_prompt_contains_stance_diversity(self):
        """Persona generator prompt requires stance diversity."""
        result = load_swarm_prompt(
            "persona_generator",
            seed_text="Test scenario",
            seed_context="",
            count="5",
        )
        assert "25%" in result
        assert "Stance Diversity" in result


def _render_convergence(**overrides) -> str:
    """Helper to render convergence prompt with defaults."""
    defaults = {
        "agent_name": "Test Agent",
        "agent_background": "Test background",
        "seed_text": "Test scenario",
        "r1_position": "support",
        "r1_confidence": "70",
        "r1_reasoning": "Test reasoning",
        "r2_summary": "Test summary",
        "debate_digest": "",
        "confidence_anchor": "medium (46-70)",
    }
    defaults.update(overrides)
    return load_swarm_prompt("convergence", **defaults)


@pytest.mark.unit
class TestConvergencePrompt:
    def test_convergence_prompt_contains_changed_mind_definition(self):
        """Convergence prompt defines changed_mind as POSITION change."""
        result = _render_convergence()
        assert "POSITION changed" in result

    def test_convergence_prompt_contains_conviction_anchor_instruction(self):
        """Convergence prompt requires specific refutation to change."""
        result = _render_convergence()
        assert "directly refutes" in result

    def test_convergence_prompt_renders_confidence_anchor(self):
        """Convergence prompt includes the confidence_anchor value."""
        result = _render_convergence(confidence_anchor="low (20-45)")
        assert "low (20-45)" in result
