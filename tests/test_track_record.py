"""Tests for track record page endpoint."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from forge.db.schema import apply_schema
from forge.export.api import create_app

if TYPE_CHECKING:
    from forge.db.store import Store


@pytest.fixture
def client(db: Store) -> TestClient:
    """FastAPI test client with cross-thread DB."""
    db.conn.close()
    db.conn = sqlite3.connect(":memory:", check_same_thread=False)
    db.conn.row_factory = sqlite3.Row
    db.conn.execute("PRAGMA foreign_keys=ON")
    apply_schema(db.conn)
    app = create_app(db)
    return TestClient(app)


class TestTrackRecord:
    def test_track_record_returns_html(self, client: TestClient) -> None:
        resp = client.get("/v1/track-record")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_track_record_contains_title(self, client: TestClient) -> None:
        resp = client.get("/v1/track-record")
        assert "FORGE Track Record" in resp.text

    def test_track_record_shows_stats(
        self, client: TestClient, db: Store,
    ) -> None:
        sim = db.save_simulation("scenario", "test")
        p1 = db.save_prediction(sim.id, "Good prediction", 80)
        p2 = db.save_prediction(sim.id, "Bad prediction", 60)
        now = datetime.now(UTC).isoformat()
        db.update_prediction(p1.id, resolved_as="true", resolved_at=now)
        db.update_prediction(p2.id, resolved_as="false", resolved_at=now)

        resp = client.get("/v1/track-record")
        assert resp.status_code == 200
        # Should show total and resolved counts
        assert "2" in resp.text  # total/resolved count

    def test_track_record_shows_recent_predictions(
        self, client: TestClient, db: Store,
    ) -> None:
        sim = db.save_simulation("scenario", "test")
        p = db.save_prediction(sim.id, "AI agents replace SaaS", 75)
        now = datetime.now(UTC).isoformat()
        db.update_prediction(p.id, resolved_as="true", resolved_at=now)

        resp = client.get("/v1/track-record")
        assert "AI agents replace SaaS" in resp.text

    def test_track_record_empty_db(self, client: TestClient) -> None:
        resp = client.get("/v1/track-record")
        assert resp.status_code == 200
        assert "No predictions" in resp.text or "0" in resp.text
