"""
Tests for forge.cli — CLI entrypoint via Typer.

Mirrors: forge/cli.py
"""


import pytest
from typer.testing import CliRunner

from forge.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# forge test "claim"
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCLITest:
    def test_test_command_exists(self):
        """'forge test' command is registered and callable."""
        result = runner.invoke(app, ["test", "--help"])
        assert result.exit_code == 0
        assert "claim" in result.output.lower() or "CLAIM" in result.output

    def test_test_command_requires_claim(self):
        """'forge test' without a claim shows an error."""
        result = runner.invoke(app, ["test"])
        assert result.exit_code != 0

    def test_test_command_json_flag_exists(self):
        """'forge test --json' flag is available."""
        result = runner.invoke(app, ["test", "--help"])
        assert "--json" in result.output


# ---------------------------------------------------------------------------
# forge status
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCLIStatus:
    def test_status_command_exists(self):
        """'forge status' command is registered."""
        result = runner.invoke(app, ["status", "--help"])
        assert result.exit_code == 0
