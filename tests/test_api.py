"""Tests for forge/export/api.py — FastAPI server."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from forge.export.api import create_app

if TYPE_CHECKING:
    from forge.db.store import Store


@pytest.fixture
def client(db: Store) -> TestClient:
    """FastAPI test client with in-memory DB."""
    # Allow cross-thread access for TestClient
    db.conn.close()
    import sqlite3

    from forge.db.schema import apply_schema

    db.conn = sqlite3.connect(":memory:", check_same_thread=False)
    db.conn.row_factory = sqlite3.Row
    db.conn.execute("PRAGMA foreign_keys=ON")
    apply_schema(db.conn)
    app = create_app(db)
    return TestClient(app)


class TestHealth:
    def test_health_endpoint(self, client: TestClient) -> None:
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestHypotheses:
    def test_list_hypotheses_empty(self, client: TestClient) -> None:
        resp = client.get("/v1/hypotheses")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_hypotheses(self, client: TestClient, db: Store) -> None:
        db.save_hypothesis("Test claim", "manual", confidence=60)
        resp = client.get("/v1/hypotheses")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["claim"] == "Test claim"

    def test_list_hypotheses_filter_status(
        self, client: TestClient, db: Store,
    ) -> None:
        db.save_hypothesis("Alive", "manual", confidence=60)
        db.save_hypothesis("Dead", "manual", status="dead")
        resp = client.get("/v1/hypotheses?status=alive")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["claim"] == "Alive"

    def test_get_hypothesis(self, client: TestClient, db: Store) -> None:
        h = db.save_hypothesis("Test claim", "manual", confidence=60)
        resp = client.get(f"/v1/hypotheses/{h.id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["claim"] == "Test claim"
        assert "relations" in data

    def test_get_hypothesis_not_found(self, client: TestClient) -> None:
        resp = client.get("/v1/hypotheses/h_nonexistent")
        assert resp.status_code == 404


class TestPredictions:
    def test_list_predictions_empty(self, client: TestClient) -> None:
        resp = client.get("/v1/predictions")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_predictions(self, client: TestClient, db: Store) -> None:
        sim = db.save_simulation("scenario", "test")
        db.save_prediction(sim.id, "BTC to 100k", 65)
        resp = client.get("/v1/predictions")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["claim"] == "BTC to 100k"


class TestFeedback:
    def test_post_feedback_endorse(
        self, client: TestClient, db: Store,
    ) -> None:
        h = db.save_hypothesis("Test", "manual")
        resp = client.post("/v1/feedback", json={
            "action": "endorse",
            "hypothesis_id": h.id,
        })
        assert resp.status_code == 200
        updated = db.get_hypothesis(h.id)
        assert updated is not None
        assert updated.human_endorsed == 1

    def test_post_feedback_reject(
        self, client: TestClient, db: Store,
    ) -> None:
        h = db.save_hypothesis("Test", "manual")
        resp = client.post("/v1/feedback", json={
            "action": "reject",
            "hypothesis_id": h.id,
        })
        assert resp.status_code == 200
        updated = db.get_hypothesis(h.id)
        assert updated is not None
        assert updated.human_rejected == 1

    def test_post_feedback_invalid_action(self, client: TestClient) -> None:
        resp = client.post("/v1/feedback", json={
            "action": "invalid",
        })
        assert resp.status_code == 400


class TestCalibration:
    def test_calibration_empty(self, client: TestClient) -> None:
        resp = client.get("/v1/calibration")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0

    def test_calibration_with_data(
        self, client: TestClient, db: Store,
    ) -> None:
        sim = db.save_simulation("scenario", "test")
        p = db.save_prediction(sim.id, "Prediction", 70)
        now = datetime.now(UTC).isoformat()
        db.update_prediction(p.id, resolved_as="true", resolved_at=now)

        resp = client.get("/v1/calibration")
        data = resp.json()
        assert data["resolved"] == 1


class TestLeaderboard:
    def test_leaderboard_empty(self, client: TestClient) -> None:
        resp = client.get("/v1/leaderboard")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_leaderboard_with_agents(
        self, client: TestClient, db: Store,
    ) -> None:
        persona_json = json.dumps({"name": "Star"})
        p = db.save_agent_persona("star", persona_json)
        db.update_agent_persona(
            p.id, calibration_score=0.85, simulations_participated=10,
        )
        resp = client.get("/v1/leaderboard")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["archetype"] == "star"


class TestBrief:
    def test_brief(self, client: TestClient) -> None:
        resp = client.get("/v1/brief")
        assert resp.status_code == 200
        data = resp.json()
        assert "generated_at" in data


class TestStats:
    def test_stats(self, client: TestClient, db: Store) -> None:
        db.save_hypothesis("Test", "manual")
        resp = client.get("/v1/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_hypotheses"] >= 1
