"""Tests for forge/analyze/prompts.py — Prompt loader and templates."""

from __future__ import annotations

import pytest

from forge.analyze.prompts import load_prompt


class TestPromptFilesExist:
    @pytest.mark.unit
    def test_steelman_prompt_loads(self) -> None:
        result = load_prompt("steelman", claim="test claim")
        assert len(result) > 100

    @pytest.mark.unit
    def test_redteam_prompt_loads(self) -> None:
        result = load_prompt("redteam", claim="test claim")
        assert len(result) > 100

    @pytest.mark.unit
    def test_judge_prompt_loads(self) -> None:
        result = load_prompt("judge", claim="test claim")
        assert len(result) > 100


class TestPromptRendering:
    @pytest.mark.unit
    def test_steelman_contains_claim(self) -> None:
        result = load_prompt("steelman", claim="AI will replace lawyers")
        assert "AI will replace lawyers" in result

    @pytest.mark.unit
    def test_redteam_contains_claim(self) -> None:
        result = load_prompt("redteam", claim="Crypto will crash")
        assert "Crypto will crash" in result

    @pytest.mark.unit
    def test_judge_contains_claim(self) -> None:
        result = load_prompt("judge", claim="Housing prices will drop")
        assert "Housing prices will drop" in result

    @pytest.mark.unit
    def test_steelman_with_context(self) -> None:
        result = load_prompt(
            "steelman",
            claim="test claim",
            context="important background info",
        )
        assert "important background info" in result

    @pytest.mark.unit
    def test_judge_with_steelman_and_redteam_args(self) -> None:
        result = load_prompt(
            "judge",
            claim="test claim",
            steelman_output="strong argument for",
            redteam_output="strong argument against",
        )
        assert "strong argument for" in result
        assert "strong argument against" in result


class TestPromptStructure:
    @pytest.mark.unit
    def test_steelman_requests_json_output(self) -> None:
        result = load_prompt("steelman", claim="test")
        assert "JSON" in result or "json" in result

    @pytest.mark.unit
    def test_redteam_requests_json_output(self) -> None:
        result = load_prompt("redteam", claim="test")
        assert "JSON" in result or "json" in result

    @pytest.mark.unit
    def test_judge_requests_json_output(self) -> None:
        result = load_prompt("judge", claim="test")
        assert "JSON" in result or "json" in result

    @pytest.mark.unit
    def test_judge_includes_confidence_instruction(self) -> None:
        result = load_prompt("judge", claim="test")
        assert "confidence" in result.lower()
        assert "0" in result and "100" in result

    @pytest.mark.unit
    def test_invalid_prompt_name_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent", claim="test")
