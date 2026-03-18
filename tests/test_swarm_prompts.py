"""
Tests for swarm prompt templates — reaction.md and persona_generator.md.

Mirrors: forge/swarm/prompts/reaction.md, forge/swarm/prompts/persona_generator.md
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
class TestPersonaGeneratorPromptConfidenceAnchor:
    def test_persona_generator_prompt_contains_confidence_anchor_field(self):
        """Persona generator prompt mentions confidence_anchor in schema."""
        result = load_swarm_prompt(
            "persona_generator",
            seed_text="Test scenario",
            seed_context="",
            count="5",
        )
        assert "confidence_anchor" in result
