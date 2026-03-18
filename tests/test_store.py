"""
Tests for forge.db — schema creation, Store CRUD, models, edge cases.

Mirrors: forge/db/schema.py, forge/db/store.py, forge/db/models.py
"""

import sqlite3

import pytest

from forge.db.models import (
    AgentPersona,
    Evidence,
    Feedback,
    Hypothesis,
    Prediction,
    Relation,
    Simulation,
    SimulationTurn,
)
from forge.db.schema import apply_schema
from forge.db.store import Store

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db() -> Store:
    """Fresh in-memory Store with schema applied."""
    store = Store(":memory:")
    return store


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSchema:
    def test_schema_creates_all_tables(self, db: Store):
        """All tables defined in FORGE-ARCHITECTURE.md Section 4 exist."""
        expected_tables = {
            "hypotheses",
            "evidence",
            "relations",
            "simulations",
            "agent_personas",
            "simulation_turns",
            "predictions",
            "feeds",
            "articles",
            "feedback",
            "calibration_snapshots",
        }
        cursor = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        actual_tables = {row[0] for row in cursor.fetchall()}
        assert expected_tables == actual_tables

    def test_schema_creates_all_indexes(self, db: Store):
        """All indexes defined in FORGE-ARCHITECTURE.md Section 4 exist."""
        expected_indexes = {
            "idx_hypotheses_status",
            "idx_hypotheses_confidence",
            "idx_hypotheses_tags",
            "idx_relations_source",
            "idx_relations_target",
            "idx_simulations_status",
            "idx_simulation_turns_sim",
            "idx_predictions_simulation",
            "idx_predictions_resolved",
            "idx_agent_personas_archetype",
            "idx_articles_url",
        }
        cursor = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
        )
        actual_indexes = {row[0] for row in cursor.fetchall()}
        assert expected_indexes.issubset(actual_indexes)

    def test_schema_apply_is_idempotent(self):
        """Calling apply_schema twice on the same connection does not raise."""
        conn = sqlite3.connect(":memory:")
        apply_schema(conn)
        apply_schema(conn)  # should not raise
        conn.close()

    def test_schema_wal_mode_on_file_db(self, tmp_path):
        """File-backed Store enables WAL journal mode."""
        db_path = str(tmp_path / "test.db")
        store = Store(db_path)
        cursor = store.conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        assert mode == "wal"
        store.conn.close()

    def test_schema_memory_db_does_not_require_wal(self, db: Store):
        """In-memory DB works without WAL (WAL is only for file-backed)."""
        # Just confirm the store is usable
        cursor = db.conn.execute("SELECT 1")
        assert cursor.fetchone()[0] == 1


# ---------------------------------------------------------------------------
# Hypothesis CRUD
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHypothesisCRUD:
    def test_store_save_hypothesis_returns_model(self, db: Store):
        """save_hypothesis returns a Hypothesis Pydantic model."""
        h = db.save_hypothesis(claim="Test claim", source="manual")
        assert isinstance(h, Hypothesis)

    def test_store_save_hypothesis_assigns_ulid(self, db: Store):
        """Saved hypothesis has an ID with h_ prefix."""
        h = db.save_hypothesis(claim="Test claim", source="manual")
        assert h.id.startswith("h_")
        assert len(h.id) > 3  # h_ + ULID

    def test_store_save_hypothesis_populates_timestamps(self, db: Store):
        """created_at and updated_at are auto-populated as ISO 8601 UTC."""
        h = db.save_hypothesis(claim="Test claim", source="manual")
        assert h.created_at is not None
        assert h.updated_at is not None
        # ISO 8601 contains 'T' separator
        assert "T" in h.created_at
        assert "T" in h.updated_at

    def test_store_save_hypothesis_default_confidence(self, db: Store):
        """Default confidence is 50."""
        h = db.save_hypothesis(claim="Test claim", source="manual")
        assert h.confidence == 50

    def test_store_save_hypothesis_default_status(self, db: Store):
        """Default status is 'alive'."""
        h = db.save_hypothesis(claim="Test claim", source="manual")
        assert h.status == "alive"

    def test_store_save_hypothesis_with_all_fields(self, db: Store):
        """Can save hypothesis with all optional fields."""
        h = db.save_hypothesis(
            claim="Full claim",
            source="manual",
            context="Some context",
            confidence=75,
            status="alive",
            tags=["ai", "pricing"],
        )
        assert h.claim == "Full claim"
        assert h.context == "Some context"
        assert h.confidence == 75
        assert h.tags == ["ai", "pricing"]

    def test_store_get_hypothesis_by_id(self, db: Store):
        """get_hypothesis retrieves by ID and returns Hypothesis model."""
        h = db.save_hypothesis(claim="Find me", source="manual")
        found = db.get_hypothesis(h.id)
        assert found is not None
        assert isinstance(found, Hypothesis)
        assert found.id == h.id
        assert found.claim == "Find me"

    def test_store_get_hypothesis_not_found(self, db: Store):
        """get_hypothesis returns None for non-existent ID."""
        found = db.get_hypothesis("h_nonexistent")
        assert found is None

    def test_store_update_hypothesis(self, db: Store):
        """update_hypothesis modifies fields and updates updated_at."""
        h = db.save_hypothesis(claim="Original", source="manual")
        original_updated = h.updated_at

        updated = db.update_hypothesis(h.id, confidence=80, status="dead")
        assert updated is not None
        assert updated.confidence == 80
        assert updated.status == "dead"
        assert updated.claim == "Original"  # unchanged field preserved
        assert updated.updated_at >= original_updated

    def test_store_update_hypothesis_not_found(self, db: Store):
        """update_hypothesis returns None for non-existent ID."""
        result = db.update_hypothesis("h_nonexistent", confidence=50)
        assert result is None

    def test_store_list_hypotheses_by_status(self, db: Store):
        """list_hypotheses filters by status."""
        db.save_hypothesis(claim="Alive one", source="manual")
        db.save_hypothesis(claim="Alive two", source="manual")
        h3 = db.save_hypothesis(claim="Dead one", source="manual")
        db.update_hypothesis(h3.id, status="dead")

        alive = db.list_hypotheses(status="alive")
        assert len(alive) == 2
        assert all(isinstance(h, Hypothesis) for h in alive)
        assert all(h.status == "alive" for h in alive)

        dead = db.list_hypotheses(status="dead")
        assert len(dead) == 1
        assert dead[0].claim == "Dead one"

    def test_store_list_hypotheses_by_confidence_range(self, db: Store):
        """list_hypotheses filters by confidence range."""
        db.save_hypothesis(claim="Low", source="manual", confidence=20)
        db.save_hypothesis(claim="Mid", source="manual", confidence=50)
        db.save_hypothesis(claim="High", source="manual", confidence=80)

        results = db.list_hypotheses(min_confidence=40, max_confidence=60)
        assert len(results) == 1
        assert results[0].claim == "Mid"

    def test_store_list_hypotheses_all(self, db: Store):
        """list_hypotheses with no filters returns all."""
        db.save_hypothesis(claim="One", source="manual")
        db.save_hypothesis(claim="Two", source="manual")
        results = db.list_hypotheses()
        assert len(results) == 2

    def test_store_list_hypotheses_empty(self, db: Store):
        """list_hypotheses on empty DB returns empty list."""
        results = db.list_hypotheses()
        assert results == []

    def test_store_save_hypothesis_missing_claim_raises(self, db: Store):
        """Saving a hypothesis without a claim raises an error."""
        with pytest.raises((ValueError, TypeError)):
            db.save_hypothesis(claim="", source="manual")

    def test_store_save_hypothesis_missing_source_raises(self, db: Store):
        """Saving a hypothesis without a source raises an error."""
        with pytest.raises((ValueError, TypeError)):
            db.save_hypothesis(claim="Valid claim", source="")


# ---------------------------------------------------------------------------
# Evidence CRUD
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEvidenceCRUD:
    def test_store_save_evidence_returns_model(self, db: Store):
        """save_evidence returns an Evidence Pydantic model."""
        e = db.save_evidence(content="Some evidence text")
        assert isinstance(e, Evidence)

    def test_store_save_evidence_assigns_ulid(self, db: Store):
        """Saved evidence has an ID with e_ prefix."""
        e = db.save_evidence(content="Evidence")
        assert e.id.startswith("e_")

    def test_store_save_evidence_populates_timestamp(self, db: Store):
        """ingested_at is auto-populated."""
        e = db.save_evidence(content="Evidence")
        assert e.ingested_at is not None
        assert "T" in e.ingested_at

    def test_store_save_evidence_with_optional_fields(self, db: Store):
        """Can save evidence with source_url and source_name."""
        e = db.save_evidence(
            content="Evidence text",
            source_url="https://example.com",
            source_name="Example",
        )
        assert e.source_url == "https://example.com"
        assert e.source_name == "Example"

    def test_store_get_evidence_by_id(self, db: Store):
        """get_evidence retrieves by ID."""
        e = db.save_evidence(content="Find this evidence")
        found = db.get_evidence(e.id)
        assert found is not None
        assert found.id == e.id
        assert found.content == "Find this evidence"

    def test_store_get_evidence_not_found(self, db: Store):
        """get_evidence returns None for non-existent ID."""
        found = db.get_evidence("e_nonexistent")
        assert found is None

    def test_store_save_evidence_empty_content_raises(self, db: Store):
        """Saving evidence with empty content raises an error."""
        with pytest.raises((ValueError, TypeError)):
            db.save_evidence(content="")


# ---------------------------------------------------------------------------
# Relation CRUD
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRelationCRUD:
    def test_store_save_relation_returns_model(self, db: Store):
        """save_relation returns a Relation Pydantic model."""
        h1 = db.save_hypothesis(claim="Claim A", source="manual")
        h2 = db.save_hypothesis(claim="Claim B", source="manual")
        r = db.save_relation(
            source_id=h1.id, target_id=h2.id, relation_type="supports"
        )
        assert isinstance(r, Relation)

    def test_store_save_relation_assigns_ulid(self, db: Store):
        """Saved relation has an ID with r_ prefix."""
        h1 = db.save_hypothesis(claim="A", source="manual")
        h2 = db.save_hypothesis(claim="B", source="manual")
        r = db.save_relation(
            source_id=h1.id, target_id=h2.id, relation_type="contradicts"
        )
        assert r.id.startswith("r_")

    def test_store_save_relation_populates_timestamp(self, db: Store):
        """created_at is auto-populated."""
        h1 = db.save_hypothesis(claim="A", source="manual")
        h2 = db.save_hypothesis(claim="B", source="manual")
        r = db.save_relation(
            source_id=h1.id, target_id=h2.id, relation_type="supports"
        )
        assert r.created_at is not None

    def test_store_save_relation_with_optional_fields(self, db: Store):
        """Can save relation with strength and reasoning."""
        h1 = db.save_hypothesis(claim="A", source="manual")
        h2 = db.save_hypothesis(claim="B", source="manual")
        r = db.save_relation(
            source_id=h1.id,
            target_id=h2.id,
            relation_type="refines",
            strength=0.8,
            reasoning="B is a refinement of A",
        )
        assert r.strength == 0.8
        assert r.reasoning == "B is a refinement of A"

    def test_store_list_relations_by_source(self, db: Store):
        """list_relations_by_source returns relations from a given source."""
        h1 = db.save_hypothesis(claim="A", source="manual")
        h2 = db.save_hypothesis(claim="B", source="manual")
        h3 = db.save_hypothesis(claim="C", source="manual")
        db.save_relation(source_id=h1.id, target_id=h2.id, relation_type="supports")
        db.save_relation(source_id=h1.id, target_id=h3.id, relation_type="contradicts")
        db.save_relation(source_id=h2.id, target_id=h3.id, relation_type="refines")

        from_h1 = db.list_relations_by_source(h1.id)
        assert len(from_h1) == 2
        assert all(isinstance(r, Relation) for r in from_h1)

    def test_store_list_relations_by_target(self, db: Store):
        """list_relations_by_target returns relations pointing to a given target."""
        h1 = db.save_hypothesis(claim="A", source="manual")
        h2 = db.save_hypothesis(claim="B", source="manual")
        h3 = db.save_hypothesis(claim="C", source="manual")
        db.save_relation(source_id=h1.id, target_id=h3.id, relation_type="supports")
        db.save_relation(source_id=h2.id, target_id=h3.id, relation_type="contradicts")

        to_h3 = db.list_relations_by_target(h3.id)
        assert len(to_h3) == 2

    def test_store_list_relations_empty(self, db: Store):
        """list_relations returns empty list when none exist."""
        h1 = db.save_hypothesis(claim="A", source="manual")
        assert db.list_relations_by_source(h1.id) == []
        assert db.list_relations_by_target(h1.id) == []


# ---------------------------------------------------------------------------
# Feedback CRUD
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFeedbackCRUD:
    def test_store_save_feedback_returns_model(self, db: Store):
        """save_feedback returns a Feedback Pydantic model."""
        h = db.save_hypothesis(claim="Test", source="manual")
        f = db.save_feedback(hypothesis_id=h.id, action="endorse")
        assert isinstance(f, Feedback)

    def test_store_save_feedback_assigns_ulid_with_prefix(self, db: Store):
        """Saved feedback has an ID with f_ prefix."""
        h = db.save_hypothesis(claim="Test", source="manual")
        f = db.save_feedback(hypothesis_id=h.id, action="reject")
        assert f.id.startswith("f_")

    def test_store_save_feedback_populates_timestamp(self, db: Store):
        """created_at is auto-populated."""
        h = db.save_hypothesis(claim="Test", source="manual")
        f = db.save_feedback(hypothesis_id=h.id, action="endorse")
        assert f.created_at is not None

    def test_store_save_feedback_with_note(self, db: Store):
        """Can save feedback with optional note."""
        h = db.save_hypothesis(claim="Test", source="manual")
        f = db.save_feedback(
            hypothesis_id=h.id,
            action="annotate",
            note="Interesting claim",
        )
        assert f.note == "Interesting claim"

    def test_store_save_feedback_with_prediction_id(self, db: Store):
        """Can save feedback referencing a prediction instead of hypothesis."""
        f = db.save_feedback(prediction_id="p_fake123", action="resolve_true")
        assert f.prediction_id == "p_fake123"


# ---------------------------------------------------------------------------
# ULID generation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestULIDGeneration:
    def test_ulid_uniqueness(self, db: Store):
        """Two hypotheses get distinct IDs."""
        h1 = db.save_hypothesis(claim="First", source="manual")
        h2 = db.save_hypothesis(claim="Second", source="manual")
        assert h1.id != h2.id

    def test_ulid_sortable_by_creation(self, db: Store):
        """IDs are sortable by creation time (ULID property)."""
        h1 = db.save_hypothesis(claim="First", source="manual")
        h2 = db.save_hypothesis(claim="Second", source="manual")
        # ULIDs sort lexicographically by time
        assert h1.id < h2.id


# ---------------------------------------------------------------------------
# Simulation CRUD
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSimulationCRUD:
    def test_store_save_simulation_returns_model(self, db: Store):
        """save_simulation returns a Simulation Pydantic model."""
        s = db.save_simulation(mode="scenario", seed_text="Test scenario")
        assert isinstance(s, Simulation)

    def test_store_save_simulation_assigns_ulid(self, db: Store):
        """Saved simulation has an ID with s_ prefix."""
        s = db.save_simulation(mode="scenario", seed_text="Test")
        assert s.id.startswith("s_")
        assert len(s.id) > 3

    def test_store_save_simulation_default_status_pending(self, db: Store):
        """Default status is 'pending'."""
        s = db.save_simulation(mode="scenario", seed_text="Test")
        assert s.status == "pending"

    def test_store_save_simulation_populates_timestamps(self, db: Store):
        """started_at is set on creation."""
        s = db.save_simulation(mode="scenario", seed_text="Test")
        assert s.started_at is not None
        assert "T" in s.started_at

    def test_store_save_simulation_with_all_fields(self, db: Store):
        """Can save simulation with all optional fields."""
        s = db.save_simulation(
            mode="claim_test",
            seed_text="The claim",
            seed_context="Some context",
            agent_count=30,
            rounds=3,
        )
        assert s.mode == "claim_test"
        assert s.seed_context == "Some context"
        assert s.agent_count == 30
        assert s.rounds == 3

    def test_store_get_simulation_by_id(self, db: Store):
        """get_simulation retrieves by ID."""
        s = db.save_simulation(mode="scenario", seed_text="Find me")
        found = db.get_simulation(s.id)
        assert found is not None
        assert found.id == s.id
        assert found.seed_text == "Find me"

    def test_store_get_simulation_not_found(self, db: Store):
        """get_simulation returns None for non-existent ID."""
        assert db.get_simulation("s_nonexistent") is None

    def test_store_update_simulation_status(self, db: Store):
        """update_simulation can change status and set completed_at."""
        s = db.save_simulation(mode="scenario", seed_text="Test")
        updated = db.update_simulation(
            s.id, status="complete", summary="Done", duration_seconds=120.5
        )
        assert updated is not None
        assert updated.status == "complete"
        assert updated.summary == "Done"
        assert updated.duration_seconds == 120.5

    def test_store_update_simulation_not_found(self, db: Store):
        """update_simulation returns None for non-existent ID."""
        assert db.update_simulation("s_nonexistent", status="failed") is None

    def test_store_list_simulations_all(self, db: Store):
        """list_simulations with no filter returns all."""
        db.save_simulation(mode="scenario", seed_text="One")
        db.save_simulation(mode="claim_test", seed_text="Two")
        results = db.list_simulations()
        assert len(results) == 2

    def test_store_list_simulations_by_status(self, db: Store):
        """list_simulations filters by status."""
        s1 = db.save_simulation(mode="scenario", seed_text="One")
        db.save_simulation(mode="scenario", seed_text="Two")
        db.update_simulation(s1.id, status="complete")
        pending = db.list_simulations(status="pending")
        assert len(pending) == 1
        assert pending[0].seed_text == "Two"

    def test_store_list_simulations_empty(self, db: Store):
        """list_simulations on empty DB returns empty list."""
        assert db.list_simulations() == []

    def test_store_save_simulation_empty_seed_text_raises(self, db: Store):
        """Saving simulation with empty seed_text raises ValueError."""
        with pytest.raises(ValueError):
            db.save_simulation(mode="scenario", seed_text="")

    def test_store_find_simulation_by_prefix_exact(self, db: Store):
        """Full ID matches via prefix search."""
        s = db.save_simulation(mode="scenario", seed_text="Test")
        found = db.find_simulation_by_prefix(s.id)
        assert found is not None
        assert found.id == s.id

    def test_store_find_simulation_by_prefix_partial(self, db: Store):
        """First 10 chars of ID resolve to the simulation."""
        s = db.save_simulation(mode="scenario", seed_text="Test")
        found = db.find_simulation_by_prefix(s.id[:10])
        assert found is not None
        assert found.id == s.id

    def test_store_find_simulation_by_prefix_no_match(self, db: Store):
        """Non-existent prefix returns None."""
        found = db.find_simulation_by_prefix("s_ZZZZZZZZZZ")
        assert found is None


# ---------------------------------------------------------------------------
# AgentPersona CRUD
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAgentPersonaCRUD:
    def test_store_save_agent_persona_returns_model(self, db: Store):
        """save_agent_persona returns an AgentPersona Pydantic model."""
        ap = db.save_agent_persona(archetype="tech_optimist", persona_json='{"name": "Alice"}')
        assert isinstance(ap, AgentPersona)

    def test_store_save_agent_persona_assigns_ulid(self, db: Store):
        """Saved persona has an ID with ap_ prefix."""
        ap = db.save_agent_persona(archetype="contrarian", persona_json='{"name": "Bob"}')
        assert ap.id.startswith("ap_")
        assert len(ap.id) > 4

    def test_store_save_agent_persona_populates_timestamps(self, db: Store):
        """created_at and updated_at are auto-populated."""
        ap = db.save_agent_persona(archetype="analyst", persona_json='{}')
        assert ap.created_at is not None
        assert ap.updated_at is not None
        assert "T" in ap.created_at

    def test_store_save_agent_persona_defaults(self, db: Store):
        """Default values: active=1, simulations_participated=0, etc."""
        ap = db.save_agent_persona(archetype="test", persona_json='{}')
        assert ap.active == 1
        assert ap.simulations_participated == 0
        assert ap.predictions_correct == 0
        assert ap.predictions_incorrect == 0

    def test_store_get_agent_persona_by_id(self, db: Store):
        """get_agent_persona retrieves by ID."""
        ap = db.save_agent_persona(archetype="hawk", persona_json='{"name": "Carol"}')
        found = db.get_agent_persona(ap.id)
        assert found is not None
        assert found.id == ap.id
        assert found.archetype == "hawk"

    def test_store_get_agent_persona_not_found(self, db: Store):
        """get_agent_persona returns None for non-existent ID."""
        assert db.get_agent_persona("ap_nonexistent") is None

    def test_store_update_agent_persona(self, db: Store):
        """update_agent_persona can increment simulations_participated."""
        ap = db.save_agent_persona(archetype="test", persona_json='{}')
        updated = db.update_agent_persona(ap.id, simulations_participated=5, calibration_score=0.75)
        assert updated is not None
        assert updated.simulations_participated == 5
        assert updated.calibration_score == 0.75

    def test_store_update_agent_persona_not_found(self, db: Store):
        """update_agent_persona returns None for non-existent ID."""
        assert db.update_agent_persona("ap_nonexistent", active=0) is None

    def test_store_list_agent_personas_all(self, db: Store):
        """list_agent_personas with no filter returns all."""
        db.save_agent_persona(archetype="a", persona_json='{}')
        db.save_agent_persona(archetype="b", persona_json='{}')
        assert len(db.list_agent_personas()) == 2

    def test_store_list_agent_personas_by_active(self, db: Store):
        """list_agent_personas filters by active status."""
        ap1 = db.save_agent_persona(archetype="a", persona_json='{}')
        db.save_agent_persona(archetype="b", persona_json='{}')
        db.update_agent_persona(ap1.id, active=0)
        active = db.list_agent_personas(active=True)
        assert len(active) == 1
        assert active[0].archetype == "b"

    def test_store_list_agent_personas_empty(self, db: Store):
        """list_agent_personas on empty DB returns empty list."""
        assert db.list_agent_personas() == []


# ---------------------------------------------------------------------------
# SimulationTurn CRUD
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSimulationTurnCRUD:
    def _setup_simulation(self, db: Store):
        """Helper: create a simulation and persona for turn tests."""
        sim = db.save_simulation(mode="scenario", seed_text="Test scenario")
        persona = db.save_agent_persona(archetype="test_agent", persona_json='{}')
        return sim, persona

    def test_store_save_simulation_turn_returns_model(self, db: Store):
        """save_simulation_turn returns a SimulationTurn model."""
        sim, persona = self._setup_simulation(db)
        turn = db.save_simulation_turn(
            simulation_id=sim.id,
            round=1,
            agent_persona_id=persona.id,
            turn_type="reaction",
            content='{"position": "support", "reasoning": "test"}',
        )
        assert isinstance(turn, SimulationTurn)

    def test_store_save_simulation_turn_assigns_ulid(self, db: Store):
        """Saved turn has an ID with st_ prefix."""
        sim, persona = self._setup_simulation(db)
        turn = db.save_simulation_turn(
            simulation_id=sim.id,
            round=1,
            agent_persona_id=persona.id,
            turn_type="reaction",
            content='{}',
        )
        assert turn.id.startswith("st_")

    def test_store_save_simulation_turn_populates_timestamp(self, db: Store):
        """created_at is auto-populated."""
        sim, persona = self._setup_simulation(db)
        turn = db.save_simulation_turn(
            simulation_id=sim.id,
            round=1,
            agent_persona_id=persona.id,
            turn_type="reaction",
            content='{}',
        )
        assert turn.created_at is not None

    def test_store_save_simulation_turn_with_optional_fields(self, db: Store):
        """Can save turn with all optional fields."""
        sim, persona = self._setup_simulation(db)
        turn = db.save_simulation_turn(
            simulation_id=sim.id,
            round=2,
            agent_persona_id=persona.id,
            turn_type="challenge",
            content='{"reasoning": "disagree"}',
            responding_to_id="st_fakeid",
            position="oppose",
            confidence=75,
            token_count=150,
        )
        assert turn.round == 2
        assert turn.turn_type == "challenge"
        assert turn.responding_to_id == "st_fakeid"
        assert turn.position == "oppose"
        assert turn.confidence == 75
        assert turn.token_count == 150

    def test_store_list_turns_by_simulation(self, db: Store):
        """list_turns_by_simulation returns all turns for a simulation."""
        sim, persona = self._setup_simulation(db)
        db.save_simulation_turn(
            simulation_id=sim.id, round=1, agent_persona_id=persona.id,
            turn_type="reaction", content='{}',
        )
        db.save_simulation_turn(
            simulation_id=sim.id, round=2, agent_persona_id=persona.id,
            turn_type="challenge", content='{}',
        )
        turns = db.list_turns_by_simulation(sim.id)
        assert len(turns) == 2
        assert all(isinstance(t, SimulationTurn) for t in turns)

    def test_store_list_turns_by_simulation_and_round(self, db: Store):
        """list_turns_by_simulation can filter by round."""
        sim, persona = self._setup_simulation(db)
        db.save_simulation_turn(
            simulation_id=sim.id, round=1, agent_persona_id=persona.id,
            turn_type="reaction", content='{}',
        )
        db.save_simulation_turn(
            simulation_id=sim.id, round=2, agent_persona_id=persona.id,
            turn_type="challenge", content='{}',
        )
        round1 = db.list_turns_by_simulation(sim.id, round=1)
        assert len(round1) == 1
        assert round1[0].round == 1

    def test_store_list_turns_by_agent(self, db: Store):
        """list_turns_by_agent returns turns for a specific agent in a simulation."""
        sim, persona1 = self._setup_simulation(db)
        persona2 = db.save_agent_persona(archetype="other", persona_json='{}')
        db.save_simulation_turn(
            simulation_id=sim.id, round=1, agent_persona_id=persona1.id,
            turn_type="reaction", content='{}',
        )
        db.save_simulation_turn(
            simulation_id=sim.id, round=1, agent_persona_id=persona2.id,
            turn_type="reaction", content='{}',
        )
        turns = db.list_turns_by_agent(sim.id, persona1.id)
        assert len(turns) == 1
        assert turns[0].agent_persona_id == persona1.id

    def test_store_list_turns_empty(self, db: Store):
        """list_turns_by_simulation returns empty list when no turns exist."""
        sim = db.save_simulation(mode="scenario", seed_text="Empty")
        assert db.list_turns_by_simulation(sim.id) == []

    def test_store_save_simulation_turn_empty_content_raises(self, db: Store):
        """Saving a turn with empty content raises ValueError."""
        sim, persona = self._setup_simulation(db)
        with pytest.raises(ValueError):
            db.save_simulation_turn(
                simulation_id=sim.id, round=1, agent_persona_id=persona.id,
                turn_type="reaction", content="",
            )

    def test_store_save_simulation_turn_with_raw_content(self, db: Store):
        """raw_content is persisted alongside parsed content."""
        sim, persona = self._setup_simulation(db)
        turn = db.save_simulation_turn(
            simulation_id=sim.id, round=1, agent_persona_id=persona.id,
            turn_type="reaction", content='{"position": "support"}',
            raw_content='```json\n{"position": "support"}\n```',
        )
        assert turn.raw_content == '```json\n{"position": "support"}\n```'

    def test_store_save_simulation_turn_raw_content_nullable(self, db: Store):
        """raw_content defaults to None when not provided."""
        sim, persona = self._setup_simulation(db)
        turn = db.save_simulation_turn(
            simulation_id=sim.id, round=1, agent_persona_id=persona.id,
            turn_type="reaction", content='{}',
        )
        assert turn.raw_content is None

    def test_store_list_turns_with_agent_returns_archetype(self, db: Store):
        """list_turns_with_agent returns turn data enriched with agent archetype."""
        sim = db.save_simulation(mode="scenario", seed_text="Test")
        import json
        persona = db.save_agent_persona(
            archetype="tech_optimist",
            persona_json=json.dumps({"name": "Alice", "archetype": "tech_optimist"}),
        )
        db.save_simulation_turn(
            simulation_id=sim.id, round=1, agent_persona_id=persona.id,
            turn_type="reaction", content='{"position": "support"}',
            position="support", confidence=80,
        )
        results = db.list_turns_with_agent(sim.id)
        assert len(results) == 1
        assert results[0]["archetype"] == "tech_optimist"
        assert results[0]["position"] == "support"

    def test_store_list_turns_with_agent_filters_by_round(self, db: Store):
        """list_turns_with_agent filters by round number."""
        sim = db.save_simulation(mode="scenario", seed_text="Test")
        persona = db.save_agent_persona(archetype="analyst", persona_json='{}')
        db.save_simulation_turn(
            simulation_id=sim.id, round=1, agent_persona_id=persona.id,
            turn_type="reaction", content='{}',
        )
        db.save_simulation_turn(
            simulation_id=sim.id, round=2, agent_persona_id=persona.id,
            turn_type="challenge", content='{}',
        )
        results = db.list_turns_with_agent(sim.id, round=1)
        assert len(results) == 1
        assert results[0]["round"] == 1

    def test_store_list_turns_with_agent_filters_by_archetype(self, db: Store):
        """list_turns_with_agent filters by archetype substring."""
        sim = db.save_simulation(mode="scenario", seed_text="Test")
        p1 = db.save_agent_persona(archetype="tech_optimist", persona_json='{}')
        p2 = db.save_agent_persona(archetype="regulatory_skeptic", persona_json='{}')
        db.save_simulation_turn(
            simulation_id=sim.id, round=1, agent_persona_id=p1.id,
            turn_type="reaction", content='{}',
        )
        db.save_simulation_turn(
            simulation_id=sim.id, round=1, agent_persona_id=p2.id,
            turn_type="reaction", content='{}',
        )
        results = db.list_turns_with_agent(sim.id, archetype="tech")
        assert len(results) == 1
        assert results[0]["archetype"] == "tech_optimist"

    def test_store_list_turns_with_agent_empty_simulation(self, db: Store):
        """list_turns_with_agent returns empty list for simulation with no turns."""
        sim = db.save_simulation(mode="scenario", seed_text="Empty")
        results = db.list_turns_with_agent(sim.id)
        assert results == []


# ---------------------------------------------------------------------------
# Prediction CRUD
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPredictionCRUD:
    def test_store_save_prediction_returns_model(self, db: Store):
        """save_prediction returns a Prediction Pydantic model."""
        sim = db.save_simulation(mode="scenario", seed_text="Test")
        p = db.save_prediction(simulation_id=sim.id, claim="Prediction A", confidence=70)
        assert isinstance(p, Prediction)

    def test_store_save_prediction_assigns_ulid(self, db: Store):
        """Saved prediction has an ID with p_ prefix."""
        sim = db.save_simulation(mode="scenario", seed_text="Test")
        p = db.save_prediction(simulation_id=sim.id, claim="Prediction", confidence=60)
        assert p.id.startswith("p_")

    def test_store_save_prediction_populates_timestamp(self, db: Store):
        """created_at is auto-populated."""
        sim = db.save_simulation(mode="scenario", seed_text="Test")
        p = db.save_prediction(simulation_id=sim.id, claim="Prediction", confidence=60)
        assert p.created_at is not None

    def test_store_save_prediction_with_optional_fields(self, db: Store):
        """Can save prediction with all optional fields."""
        sim = db.save_simulation(mode="scenario", seed_text="Test")
        p = db.save_prediction(
            simulation_id=sim.id,
            claim="Specific prediction",
            confidence=80,
            consensus_strength=0.85,
            dissent_summary="Some disagreed",
            resolution_deadline="2026-06-01T00:00:00+00:00",
        )
        assert p.consensus_strength == 0.85
        assert p.dissent_summary == "Some disagreed"
        assert p.resolution_deadline == "2026-06-01T00:00:00+00:00"

    def test_store_get_prediction_by_id(self, db: Store):
        """get_prediction retrieves by ID."""
        sim = db.save_simulation(mode="scenario", seed_text="Test")
        p = db.save_prediction(simulation_id=sim.id, claim="Find me", confidence=50)
        found = db.get_prediction(p.id)
        assert found is not None
        assert found.id == p.id
        assert found.claim == "Find me"

    def test_store_get_prediction_not_found(self, db: Store):
        """get_prediction returns None for non-existent ID."""
        assert db.get_prediction("p_nonexistent") is None

    def test_store_update_prediction_resolution(self, db: Store):
        """update_prediction can set resolution fields."""
        sim = db.save_simulation(mode="scenario", seed_text="Test")
        p = db.save_prediction(simulation_id=sim.id, claim="Will resolve", confidence=70)
        updated = db.update_prediction(
            p.id,
            resolved_as="true",
            resolution_evidence="Confirmed by data",
        )
        assert updated is not None
        assert updated.resolved_as == "true"
        assert updated.resolution_evidence == "Confirmed by data"

    def test_store_update_prediction_not_found(self, db: Store):
        """update_prediction returns None for non-existent ID."""
        assert db.update_prediction("p_nonexistent", resolved_as="false") is None

    def test_store_list_predictions_by_simulation(self, db: Store):
        """list_predictions filters by simulation_id."""
        sim1 = db.save_simulation(mode="scenario", seed_text="One")
        sim2 = db.save_simulation(mode="scenario", seed_text="Two")
        db.save_prediction(simulation_id=sim1.id, claim="P1", confidence=60)
        db.save_prediction(simulation_id=sim1.id, claim="P2", confidence=70)
        db.save_prediction(simulation_id=sim2.id, claim="P3", confidence=50)
        results = db.list_predictions(simulation_id=sim1.id)
        assert len(results) == 2

    def test_store_list_predictions_all(self, db: Store):
        """list_predictions with no filter returns all."""
        sim = db.save_simulation(mode="scenario", seed_text="Test")
        db.save_prediction(simulation_id=sim.id, claim="A", confidence=50)
        db.save_prediction(simulation_id=sim.id, claim="B", confidence=60)
        assert len(db.list_predictions()) == 2

    def test_store_list_predictions_empty(self, db: Store):
        """list_predictions on empty DB returns empty list."""
        assert db.list_predictions() == []

    def test_store_save_prediction_empty_claim_raises(self, db: Store):
        """Saving prediction with empty claim raises ValueError."""
        sim = db.save_simulation(mode="scenario", seed_text="Test")
        with pytest.raises(ValueError):
            db.save_prediction(simulation_id=sim.id, claim="", confidence=50)
