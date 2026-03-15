"""
Tests for forge prompt loading and rendering.

Mirrors: forge/analyze/prompts.py (prompt loader)
         forge/analyze/prompts/ (prompt template files)
"""

import pytest

from forge.analyze.prompts import PROMPTS_DIR, load_prompt

# ---------------------------------------------------------------------------
# Prompt files exist
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPromptFilesExist:
    def test_steelman_prompt_exists(self):
        """steelman.md prompt file exists."""
        path = PROMPTS_DIR / "steelman.md"
        assert path.exists(), f"Missing prompt file: {path}"

    def test_redteam_prompt_exists(self):
        """redteam.md prompt file exists."""
        path = PROMPTS_DIR / "redteam.md"
        assert path.exists(), f"Missing prompt file: {path}"

    def test_judge_prompt_exists(self):
        """judge.md prompt file exists."""
        path = PROMPTS_DIR / "judge.md"
        assert path.exists(), f"Missing prompt file: {path}"


# ---------------------------------------------------------------------------
# Prompt loading and rendering
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPromptLoading:
    def test_load_prompt_returns_string(self):
        """load_prompt returns a non-empty string."""
        result = load_prompt("steelman", claim="Test claim")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_load_prompt_renders_claim_variable(self):
        """Rendered prompt contains the claim text."""
        result = load_prompt("steelman", claim="AI will replace all jobs")
        assert "AI will replace all jobs" in result

    def test_load_prompt_renders_context_variable(self):
        """Rendered prompt contains the context when provided."""
        result = load_prompt(
            "steelman",
            claim="Test claim",
            context="Background info here",
        )
        assert "Background info here" in result

    def test_load_prompt_without_context(self):
        """Prompt renders cleanly when context is not provided."""
        result = load_prompt("steelman", claim="Test claim")
        assert "Test claim" in result

    def test_load_prompt_invalid_name_raises(self):
        """Loading a non-existent prompt raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent", claim="test")


# ---------------------------------------------------------------------------
# Prompt content structure
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPromptContent:
    def test_steelman_prompt_has_role_definition(self):
        """Steelman prompt defines the role."""
        result = load_prompt("steelman", claim="test")
        assert "steelman" in result.lower() or "strongest case" in result.lower()

    def test_redteam_prompt_has_role_definition(self):
        """Redteam prompt defines the adversarial role."""
        result = load_prompt("redteam", claim="test")
        lower = result.lower()
        assert "red team" in lower or "attack" in lower or "challenge" in lower

    def test_judge_prompt_has_role_definition(self):
        """Judge prompt defines the synthesis role."""
        result = load_prompt("judge", claim="test")
        assert "judge" in result.lower() or "synthe" in result.lower()

    def test_judge_prompt_mentions_confidence(self):
        """Judge prompt instructs confidence scoring 0-100."""
        result = load_prompt("judge", claim="test")
        assert "confidence" in result.lower()

    def test_all_prompts_request_json_output(self):
        """All prompts instruct the LLM to output JSON."""
        for name in ("steelman", "redteam", "judge"):
            result = load_prompt(name, claim="test")
            assert "json" in result.lower(), f"{name} prompt doesn't mention JSON output"

    def test_judge_prompt_accepts_steelman_and_redteam_args(self):
        """Judge prompt can render with steelman_arg and redteam_arg."""
        result = load_prompt(
            "judge",
            claim="test claim",
            steelman_arg="strong case here",
            redteam_arg="weak points here",
        )
        assert "strong case here" in result
        assert "weak points here" in result
