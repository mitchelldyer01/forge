"""
Tests for `forge turns` CLI command.

Mirrors: forge/cli.py (turns command)
"""

import json

import pytest
from typer.testing import CliRunner

from forge.cli import app
from forge.db.store import Store

runner = CliRunner()


@pytest.fixture
def db() -> Store:
    """Fresh in-memory Store."""
    return Store(":memory:")


def _patch_store(monkeypatch, db: Store):
    """Patch _get_store to return our in-memory store."""
    monkeypatch.setattr("forge.cli._get_store", lambda: db)


@pytest.mark.unit
class TestTurnsCommand:
    def test_turns_no_args_lists_simulations(self, db: Store, monkeypatch):
        """forge turns with no simulation_id lists recent simulations."""
        _patch_store(monkeypatch, db)
        db.save_simulation(
            mode="scenario", seed_text="EU AI regulation", agent_count=10, rounds=3,
        )
        result = runner.invoke(app, ["turns"])
        assert result.exit_code == 0
        assert "Simulations" in result.output

    def test_turns_no_simulations_shows_message(self, db: Store, monkeypatch):
        """forge turns with no simulations shows empty message."""
        _patch_store(monkeypatch, db)
        result = runner.invoke(app, ["turns"])
        assert result.exit_code == 0
        assert "No simulations" in result.output

    def test_turns_with_sim_id_shows_turns(self, db: Store, monkeypatch):
        """forge turns <sim_id> shows turns table."""
        _patch_store(monkeypatch, db)
        sim = db.save_simulation(mode="scenario", seed_text="Test")
        persona = db.save_agent_persona(
            archetype="tech_optimist",
            persona_json=json.dumps({"name": "Alice", "archetype": "tech_optimist"}),
        )
        db.save_simulation_turn(
            simulation_id=sim.id, round=1, agent_persona_id=persona.id,
            turn_type="reaction",
            content='{"position": "support", "reasoning": "Innovation drives growth"}',
            position="support", confidence=80,
        )
        result = runner.invoke(app, ["turns", sim.id])
        assert result.exit_code == 0
        assert "tech_optimist" in result.output
        assert "support" in result.output

    def test_turns_round_filter(self, db: Store, monkeypatch):
        """--round filters to specific round."""
        _patch_store(monkeypatch, db)
        sim = db.save_simulation(mode="scenario", seed_text="Test")
        persona = db.save_agent_persona(archetype="analyst", persona_json='{}')
        db.save_simulation_turn(
            simulation_id=sim.id, round=1, agent_persona_id=persona.id,
            turn_type="reaction", content='{}', position="support",
        )
        db.save_simulation_turn(
            simulation_id=sim.id, round=2, agent_persona_id=persona.id,
            turn_type="challenge", content='{}', position="oppose",
        )
        result = runner.invoke(app, ["turns", sim.id, "--round", "1"])
        assert result.exit_code == 0
        assert "reaction" in result.output

    def test_turns_detail_mode(self, db: Store, monkeypatch):
        """--detail shows full content."""
        _patch_store(monkeypatch, db)
        sim = db.save_simulation(mode="scenario", seed_text="Test")
        persona = db.save_agent_persona(archetype="analyst", persona_json='{}')
        db.save_simulation_turn(
            simulation_id=sim.id, round=1, agent_persona_id=persona.id,
            turn_type="reaction",
            content='{"position": "support", "reasoning": "Very detailed reasoning here"}',
            position="support", confidence=90,
        )
        result = runner.invoke(app, ["turns", sim.id, "--detail"])
        assert result.exit_code == 0
        assert "Very detailed reasoning here" in result.output

    def test_turns_json_output(self, db: Store, monkeypatch):
        """--json outputs valid JSON."""
        _patch_store(monkeypatch, db)
        sim = db.save_simulation(mode="scenario", seed_text="Test")
        persona = db.save_agent_persona(archetype="analyst", persona_json='{}')
        db.save_simulation_turn(
            simulation_id=sim.id, round=1, agent_persona_id=persona.id,
            turn_type="reaction", content='{"position": "support"}',
            position="support", confidence=70,
        )
        result = runner.invoke(app, ["turns", sim.id, "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["archetype"] == "analyst"

    def test_turns_agent_filter(self, db: Store, monkeypatch):
        """--agent filters by archetype substring."""
        _patch_store(monkeypatch, db)
        sim = db.save_simulation(mode="scenario", seed_text="Test")
        p1 = db.save_agent_persona(archetype="tech_optimist", persona_json='{}')
        p2 = db.save_agent_persona(archetype="regulatory_skeptic", persona_json='{}')
        db.save_simulation_turn(
            simulation_id=sim.id, round=1, agent_persona_id=p1.id,
            turn_type="reaction", content='{}', position="support",
        )
        db.save_simulation_turn(
            simulation_id=sim.id, round=1, agent_persona_id=p2.id,
            turn_type="reaction", content='{}', position="oppose",
        )
        result = runner.invoke(app, ["turns", sim.id, "--agent", "tech"])
        assert result.exit_code == 0
        assert "tech_optimist" in result.output
        # regulatory_skeptic should be filtered out
        assert "regulatory_skeptic" not in result.output

    def test_turns_invalid_sim_id(self, db: Store, monkeypatch):
        """forge turns with invalid sim_id shows error."""
        _patch_store(monkeypatch, db)
        result = runner.invoke(app, ["turns", "s_nonexistent"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_turns_prefix_match(self, db: Store, monkeypatch):
        """Partial simulation ID prefix resolves to full ID."""
        _patch_store(monkeypatch, db)
        sim = db.save_simulation(mode="scenario", seed_text="Test")
        persona = db.save_agent_persona(archetype="analyst", persona_json='{}')
        db.save_simulation_turn(
            simulation_id=sim.id, round=1, agent_persona_id=persona.id,
            turn_type="reaction", content='{"position": "support"}',
            position="support", confidence=70,
        )
        # Use only first 10 chars of the ID
        result = runner.invoke(app, ["turns", sim.id[:10]])
        assert result.exit_code == 0
        assert "analyst" in result.output
