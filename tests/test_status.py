"""
Tests for forge status CLI command.

Mirrors: forge/cli.py (status subcommand)
"""

import pytest
from typer.testing import CliRunner

from forge.cli import app
from forge.db.store import Store

runner = CliRunner()


# ---------------------------------------------------------------------------
# forge status
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestStatusCommand:
    def test_status_command_exists(self):
        """'forge status' command is registered."""
        result = runner.invoke(app, ["status", "--help"])
        assert result.exit_code == 0

    def test_status_shows_db_info(self, tmp_path, monkeypatch):
        """Status shows database information."""
        db_path = str(tmp_path / "test.db")
        Store(db_path)  # Create the DB
        monkeypatch.setenv("FORGE_DB_PATH", db_path)
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "DB Path" in result.output

    def test_status_shows_hypothesis_counts(self, tmp_path, monkeypatch):
        """Status shows hypothesis count by status."""
        db_path = str(tmp_path / "test.db")
        store = Store(db_path)
        store.save_hypothesis(claim="Alive", source="manual")
        h = store.save_hypothesis(claim="Dead", source="manual")
        store.update_hypothesis(h.id, status="dead")
        store.conn.close()

        monkeypatch.setenv("FORGE_DB_PATH", db_path)
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "1" in result.output

    def test_status_shows_llm_url(self, tmp_path, monkeypatch):
        """Status shows the configured LLM URL."""
        db_path = str(tmp_path / "test.db")
        Store(db_path)
        monkeypatch.setenv("FORGE_DB_PATH", db_path)
        monkeypatch.setenv("FORGE_LLAMA_URL", "http://localhost:9999")
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "localhost:9999" in result.output

    def test_status_on_empty_db(self, tmp_path, monkeypatch):
        """Status works on an empty database."""
        db_path = str(tmp_path / "empty.db")
        monkeypatch.setenv("FORGE_DB_PATH", db_path)
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "0" in result.output
