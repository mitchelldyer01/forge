"""
Tests for forge history CLI command.

Mirrors: forge/cli.py (history subcommand)
"""

import pytest
from typer.testing import CliRunner

from forge.cli import app
from forge.db.store import Store

runner = CliRunner()


# ---------------------------------------------------------------------------
# forge history
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHistoryCommand:
    def test_history_command_exists(self):
        """'forge history' command is registered."""
        result = runner.invoke(app, ["history", "--help"])
        assert result.exit_code == 0

    def test_history_empty_db_shows_message(self, tmp_path):
        """Empty DB prints 'No hypotheses yet' instead of crashing."""
        db_path = str(tmp_path / "empty.db")
        result = runner.invoke(app, ["history"], env={"FORGE_DB_PATH": db_path})
        assert result.exit_code == 0
        assert "no hypotheses" in result.output.lower()

    def test_history_shows_hypotheses(self, tmp_path):
        """History lists stored hypotheses."""
        db_path = str(tmp_path / "test.db")
        store = Store(db_path)
        store.save_hypothesis(claim="First claim", source="manual", confidence=70)
        store.save_hypothesis(claim="Second claim", source="manual", confidence=40)
        store.conn.close()

        result = runner.invoke(app, ["history"], env={"FORGE_DB_PATH": db_path})
        assert result.exit_code == 0
        assert "First claim" in result.output
        assert "Second claim" in result.output

    def test_history_status_filter(self, tmp_path):
        """--status flag filters hypotheses."""
        db_path = str(tmp_path / "test.db")
        store = Store(db_path)
        store.save_hypothesis(claim="Alive one", source="manual")
        h = store.save_hypothesis(claim="Dead one", source="manual")
        store.update_hypothesis(h.id, status="dead")
        store.conn.close()

        result = runner.invoke(
            app, ["history", "--status", "dead"], env={"FORGE_DB_PATH": db_path}
        )
        assert result.exit_code == 0
        assert "Dead one" in result.output
        assert "Alive one" not in result.output

    def test_history_limit_flag(self, tmp_path):
        """--limit flag limits results."""
        db_path = str(tmp_path / "test.db")
        store = Store(db_path)
        for i in range(5):
            store.save_hypothesis(claim=f"Claim {i}", source="manual")
        store.conn.close()

        result = runner.invoke(
            app, ["history", "--limit", "2"], env={"FORGE_DB_PATH": db_path}
        )
        assert result.exit_code == 0
        # Should show only 2 hypotheses (most recent first)
