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


class TestForgeHistory:
    @pytest.mark.integration
    def test_history_empty_db(self, tmp_path) -> None:
        with patch("forge.cli._get_store") as mock_store:
            from forge.db.store import Store

            store = Store(":memory:")
            mock_store.return_value = store
            result = runner.invoke(app, ["history"])
            assert result.exit_code == 0
            assert "No hypotheses yet" in result.output

    @pytest.mark.integration
    def test_history_shows_hypotheses(self, tmp_path) -> None:
        with patch("forge.cli._get_store") as mock_store:
            from forge.db.store import Store

            store = Store(":memory:")
            store.save_hypothesis(claim="Test claim one", source="manual", confidence=75)
            store.save_hypothesis(claim="Test claim two", source="manual", confidence=30)
            mock_store.return_value = store
            result = runner.invoke(app, ["history"])
            assert result.exit_code == 0
            assert "Test claim one" in result.output
            assert "Test claim two" in result.output

    @pytest.mark.integration
    def test_history_status_filter(self) -> None:
        with patch("forge.cli._get_store") as mock_store:
            from forge.db.store import Store

            store = Store(":memory:")
            store.save_hypothesis(claim="Alive claim", source="manual")
            h = store.save_hypothesis(claim="Dead claim", source="manual")
            store.update_hypothesis(h.id, status="dead")
            mock_store.return_value = store
            result = runner.invoke(app, ["history", "--status", "alive"])
            assert result.exit_code == 0
            assert "Alive claim" in result.output
            assert "Dead claim" not in result.output

    @pytest.mark.integration
    def test_history_limit(self) -> None:
        with patch("forge.cli._get_store") as mock_store:
            from forge.db.store import Store

            store = Store(":memory:")
            for i in range(5):
                store.save_hypothesis(claim=f"Hypothesis {i}", source="manual")
            mock_store.return_value = store
            result = runner.invoke(app, ["history", "--limit", "2"])
            assert result.exit_code == 0
            # Should only show 2 hypotheses (table header also says "Hypotheses")
            hypothesis_count = sum(
                1 for i in range(5) if f"Hypothesis {i}" in result.output
            )
            assert hypothesis_count == 2


class TestForgeStatus:
    @pytest.mark.integration
    def test_status_shows_db_stats(self) -> None:
        with patch("forge.cli._get_store") as mock_store:
            from forge.db.store import Store

            store = Store(":memory:")
            store.save_hypothesis(claim="Test", source="manual")
            store.save_evidence(content="Some evidence")
            store.save_feedback(action="endorse", hypothesis_id="h_fake")
            mock_store.return_value = store

            with patch("forge.cli._check_llm_health", return_value=("healthy", "qwen3")):
                result = runner.invoke(app, ["status"])
            assert result.exit_code == 0
            assert "1" in result.output  # hypothesis count

    @pytest.mark.integration
    def test_status_llm_unreachable(self) -> None:
        with patch("forge.cli._get_store") as mock_store:
            from forge.db.store import Store

            mock_store.return_value = Store(":memory:")

            with patch("forge.cli._check_llm_health", return_value=("unreachable", None)):
                result = runner.invoke(app, ["status"])
            assert result.exit_code == 0
            assert "unreachable" in result.output.lower()
