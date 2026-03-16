"""Tests for forge/cli.py — CLI entrypoint."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from forge.cli import app

runner = CliRunner()


def _mock_analyze_result():
    """Create a mock Verdict for testing."""
    from forge.analyze.structured import Verdict

    return Verdict(
        position="AI agents will partially displace SaaS",
        confidence=42,
        synthesis="Both sides have valid points",
        steelman_arg="AI can replace SaaS tools",
        redteam_arg="Enterprise switching costs are high",
        conditions=["Faster if cloud providers bundle AI"],
        tags=["ai", "saas"],
    )


class TestForgeTest:
    @pytest.mark.integration
    def test_forge_test_runs_analysis(self) -> None:
        with patch("forge.cli.analyze", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = _mock_analyze_result()
            result = runner.invoke(app, ["test", "AI will replace SaaS"])
            assert result.exit_code == 0
            assert "42" in result.output  # confidence score
            mock_analyze.assert_called_once()

    @pytest.mark.integration
    def test_forge_test_with_context(self) -> None:
        with patch("forge.cli.analyze", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = _mock_analyze_result()
            result = runner.invoke(
                app, ["test", "AI will replace SaaS", "--context", "Based on trends"]
            )
            assert result.exit_code == 0
            # Verify context was passed
            call_kwargs = mock_analyze.call_args
            assert call_kwargs.kwargs.get("context") == "Based on trends"

    @pytest.mark.integration
    def test_forge_test_json_output(self) -> None:
        with patch("forge.cli.analyze", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = _mock_analyze_result()
            result = runner.invoke(app, ["test", "test claim", "--json"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["confidence"] == 42
            assert data["position"] == "AI agents will partially displace SaaS"

    @pytest.mark.integration
    def test_forge_test_nonzero_exit_on_failure(self) -> None:
        with patch("forge.cli.analyze", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.side_effect = Exception("LLM down")
            result = runner.invoke(app, ["test", "test claim"])
            assert result.exit_code != 0
