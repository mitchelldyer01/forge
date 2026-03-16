"""Tests for forge/cli.py — CLI entrypoint."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from forge.cli import app
from forge.db.models import Prediction, Simulation
from forge.swarm.arena import SimulationResult
from forge.swarm.consensus import ConsensusReport

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


class TestForgeGraph:
    @pytest.mark.integration
    def test_graph_shows_hypothesis_and_relations(self) -> None:
        with patch("forge.cli._get_store") as mock_store:
            from forge.db.store import Store

            store = Store(":memory:")
            h1 = store.save_hypothesis(claim="AI will grow", source="manual")
            h2 = store.save_hypothesis(claim="SaaS will decline", source="manual")
            store.save_relation(h1.id, h2.id, "supports", reasoning="AI replaces SaaS")
            mock_store.return_value = store

            result = runner.invoke(app, ["graph", h1.id])
            assert result.exit_code == 0
            assert "AI will grow" in result.output
            assert "SaaS will decline" in result.output
            assert "supports" in result.output

    @pytest.mark.integration
    def test_graph_hypothesis_not_found(self) -> None:
        with patch("forge.cli._get_store") as mock_store:
            from forge.db.store import Store

            mock_store.return_value = Store(":memory:")
            result = runner.invoke(app, ["graph", "h_nonexistent"])
            assert result.exit_code == 0
            assert "not found" in result.output.lower()

    @pytest.mark.integration
    def test_graph_no_relations(self) -> None:
        with patch("forge.cli._get_store") as mock_store:
            from forge.db.store import Store

            store = Store(":memory:")
            h = store.save_hypothesis(claim="Standalone claim", source="manual")
            mock_store.return_value = store

            result = runner.invoke(app, ["graph", h.id])
            assert result.exit_code == 0
            assert "Standalone claim" in result.output
            assert "No relations" in result.output


def _mock_simulate_result():
    """Create mock simulation result for testing."""
    sim = Simulation(
        id="s_test123",
        mode="scenario",
        seed_text="Test scenario",
        agent_count=5,
        rounds=3,
        status="complete",
        predictions_extracted=1,
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:02:00+00:00",
        duration_seconds=120.0,
    )
    consensus = ConsensusReport(
        majority_position="support",
        majority_confidence=72.0,
        majority_fraction=0.7,
    )
    prediction = Prediction(
        id="p_test123",
        simulation_id="s_test123",
        claim="Test prediction claim",
        confidence=65,
        consensus_strength=0.8,
        created_at="2026-01-01T00:02:00+00:00",
    )
    sim_result = SimulationResult(
        simulation=sim,
        turns=[],
        consensus=consensus,
        duration_seconds=120.0,
    )
    return {
        "simulation": sim,
        "sim_result": sim_result,
        "consensus": consensus,
        "predictions": [prediction],
    }


class TestForgeSimulate:
    def _patch_simulate(self):
        """Patch _get_store, _get_llm, and _run_simulate for simulate tests."""
        from forge.db.store import Store

        store = Store(":memory:")
        return (
            patch("forge.cli._get_store", return_value=store),
            patch("forge.cli._get_llm", return_value=MagicMock()),
            patch("forge.cli._run_simulate", new_callable=AsyncMock),
        )

    @pytest.mark.integration
    def test_simulate_runs_successfully(self) -> None:
        p_store, p_llm, p_run = self._patch_simulate()
        with p_store, p_llm, p_run as mock_run:
            mock_run.return_value = _mock_simulate_result()
            result = runner.invoke(app, ["simulate", "Test scenario"])
            assert result.exit_code == 0
            assert "FORGE Simulation" in result.output
            assert "support" in result.output
            assert "Test prediction claim" in result.output

    @pytest.mark.integration
    def test_simulate_json_output(self) -> None:
        p_store, p_llm, p_run = self._patch_simulate()
        with p_store, p_llm, p_run as mock_run:
            mock_run.return_value = _mock_simulate_result()
            result = runner.invoke(app, ["simulate", "Test scenario", "--json"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["majority_position"] == "support"
            assert data["majority_confidence"] == 72.0
            assert len(data["predictions"]) == 1

    @pytest.mark.integration
    def test_simulate_nonzero_exit_on_failure(self) -> None:
        p_store, p_llm, p_run = self._patch_simulate()
        with p_store, p_llm, p_run as mock_run:
            mock_run.side_effect = Exception("LLM down")
            result = runner.invoke(app, ["simulate", "Test scenario"])
            assert result.exit_code != 0

    @pytest.mark.integration
    def test_simulate_passes_options(self) -> None:
        p_store, p_llm, p_run = self._patch_simulate()
        with p_store, p_llm, p_run as mock_run:
            mock_run.return_value = _mock_simulate_result()
            result = runner.invoke(
                app,
                ["simulate", "Test", "--agents", "10", "--rounds", "2", "--context", "BG"],
            )
            assert result.exit_code == 0
            call_args = mock_run.call_args
            assert call_args[0][3] == 10  # agent_count
            assert call_args[0][4] == 2   # round_count
