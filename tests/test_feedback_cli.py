"""Tests for Phase 4 CLI commands: endorse, reject, brief, leaderboard, evolve, export."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import patch

from typer.testing import CliRunner

from forge.cli import app

if TYPE_CHECKING:
    from forge.db.store import Store

runner = CliRunner()


class TestEndorse:
    def test_endorse_increments_counter(self, db: Store) -> None:
        h = db.save_hypothesis("Test claim", "manual", confidence=50)
        with patch("forge.cli._get_store", return_value=db):
            result = runner.invoke(app, ["endorse", h.id])
        assert result.exit_code == 0
        updated = db.get_hypothesis(h.id)
        assert updated is not None
        assert updated.human_endorsed == 1

    def test_endorse_multiple_times(self, db: Store) -> None:
        h = db.save_hypothesis("Test claim", "manual", confidence=50)
        with patch("forge.cli._get_store", return_value=db):
            runner.invoke(app, ["endorse", h.id])
            runner.invoke(app, ["endorse", h.id])
        updated = db.get_hypothesis(h.id)
        assert updated is not None
        assert updated.human_endorsed == 2

    def test_endorse_nonexistent(self, db: Store) -> None:
        with patch("forge.cli._get_store", return_value=db):
            result = runner.invoke(app, ["endorse", "h_nonexistent"])
        assert result.exit_code == 1

    def test_endorse_with_note(self, db: Store) -> None:
        h = db.save_hypothesis("Test claim", "manual")
        with patch("forge.cli._get_store", return_value=db):
            result = runner.invoke(
                app, ["endorse", h.id, "--note", "Looks right"],
            )
        assert result.exit_code == 0


class TestReject:
    def test_reject_increments_counter(self, db: Store) -> None:
        h = db.save_hypothesis("Bad claim", "manual", confidence=50)
        with patch("forge.cli._get_store", return_value=db):
            result = runner.invoke(app, ["reject", h.id])
        assert result.exit_code == 0
        updated = db.get_hypothesis(h.id)
        assert updated is not None
        assert updated.human_rejected == 1

    def test_reject_nonexistent(self, db: Store) -> None:
        with patch("forge.cli._get_store", return_value=db):
            result = runner.invoke(app, ["reject", "h_nonexistent"])
        assert result.exit_code == 1


class TestBrief:
    def test_brief_renders_digest(self, db: Store) -> None:
        db.save_hypothesis("Test claim", "manual", confidence=50)
        with patch("forge.cli._get_store", return_value=db):
            result = runner.invoke(app, ["brief"])
        assert result.exit_code == 0
        assert "FORGE Daily Digest" in result.output

    def test_brief_empty_db(self, db: Store) -> None:
        with patch("forge.cli._get_store", return_value=db):
            result = runner.invoke(app, ["brief"])
        assert result.exit_code == 0


class TestLeaderboard:
    def test_leaderboard_shows_agents(self, db: Store) -> None:
        persona_json = json.dumps({"name": "Star", "background": "test"})
        p = db.save_agent_persona("star_analyst", persona_json)
        db.update_agent_persona(
            p.id, calibration_score=0.85, simulations_participated=10,
        )
        with patch("forge.cli._get_store", return_value=db):
            result = runner.invoke(app, ["leaderboard"])
        assert result.exit_code == 0
        assert "star_analyst" in result.output

    def test_leaderboard_empty(self, db: Store) -> None:
        with patch("forge.cli._get_store", return_value=db):
            result = runner.invoke(app, ["leaderboard"])
        assert result.exit_code == 0
        assert "No scored agents" in result.output


class TestEvolve:
    def test_evolve_runs_cycle(self, db: Store) -> None:
        h = db.save_hypothesis("weak claim", "manual", confidence=15)
        old_date = (datetime.now(UTC) - timedelta(days=10)).isoformat()
        db.update_hypothesis(h.id, created_at=old_date)

        with patch("forge.cli._get_store", return_value=db):
            result = runner.invoke(app, ["evolve"])
        assert result.exit_code == 0
        assert "culled" in result.output.lower() or "evolution" in result.output.lower()


class TestExport:
    def test_export_obsidian(self, db: Store, tmp_path) -> None:
        db.save_hypothesis("Test claim", "manual", confidence=50)
        vault_path = str(tmp_path / "vault")
        with patch("forge.cli._get_store", return_value=db):
            result = runner.invoke(app, ["export", vault_path])
        assert result.exit_code == 0
        assert (tmp_path / "vault" / "hypotheses").exists()
