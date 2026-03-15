"""
Tests for forge.db — schema creation, Store CRUD, models, edge cases.

Mirrors: forge/db/schema.py, forge/db/store.py, forge/db/models.py
"""

import sqlite3

import pytest

from forge.db.models import (
    Evidence,
    Feedback,
    Hypothesis,
    Relation,
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
