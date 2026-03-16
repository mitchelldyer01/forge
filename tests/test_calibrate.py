"""Tests for calibration: resolution tracking, scoring, drift detection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from forge.db.models import Prediction, Simulation
    from forge.db.store import Store


# ------------------------------------------------------------------
# Resolution tracking
# ------------------------------------------------------------------


@pytest.mark.unit
class TestResolutionTracking:
    def test_resolve_prediction_as_true(
        self, db: Store, sample_prediction: Prediction,
    ) -> None:
        from forge.calibrate.resolver import resolve_prediction

        result = resolve_prediction(
            db, sample_prediction.id, "true", note="Confirmed by data",
        )
        assert result.resolved_as == "true"
        assert result.resolved_at is not None
        assert result.resolution_evidence is None

    def test_resolve_prediction_as_false(
        self, db: Store, sample_prediction: Prediction,
    ) -> None:
        from forge.calibrate.resolver import resolve_prediction

        result = resolve_prediction(
            db, sample_prediction.id, "false",
            evidence="Counter-evidence found",
        )
        assert result.resolved_as == "false"
        assert result.resolution_evidence == "Counter-evidence found"

    def test_resolve_prediction_as_partial(
        self, db: Store, sample_prediction: Prediction,
    ) -> None:
        from forge.calibrate.resolver import resolve_prediction

        result = resolve_prediction(db, sample_prediction.id, "partial")
        assert result.resolved_as == "partial"

    def test_resolve_prediction_saves_feedback(
        self, db: Store, sample_prediction: Prediction,
    ) -> None:
        from forge.calibrate.resolver import resolve_prediction

        resolve_prediction(db, sample_prediction.id, "true", note="Verified")
        feedback_count = db.count_feedback()
        assert feedback_count >= 1

    def test_resolve_prediction_invalid_outcome_raises(
        self, db: Store, sample_prediction: Prediction,
    ) -> None:
        from forge.calibrate.resolver import resolve_prediction

        with pytest.raises(ValueError, match="Invalid outcome"):
            resolve_prediction(db, sample_prediction.id, "maybe")

    def test_resolve_prediction_not_found_raises(self, db: Store) -> None:
        from forge.calibrate.resolver import resolve_prediction

        with pytest.raises(ValueError, match="not found"):
            resolve_prediction(db, "p_nonexistent", "true")

    def test_check_overdue_predictions_returns_past_deadline(
        self, db: Store, sample_simulation: Simulation,
    ) -> None:
        from forge.calibrate.resolver import check_overdue_predictions

        past = (datetime.now(UTC) - timedelta(days=30)).isoformat()
        db.save_prediction(
            simulation_id=sample_simulation.id,
            claim="Past deadline prediction",
            confidence=60,
            resolution_deadline=past,
        )

        overdue = check_overdue_predictions(db)
        assert len(overdue) == 1
        assert overdue[0].claim == "Past deadline prediction"

    def test_check_overdue_skips_no_deadline(
        self, db: Store, sample_simulation: Simulation,
    ) -> None:
        from forge.calibrate.resolver import check_overdue_predictions

        db.save_prediction(
            simulation_id=sample_simulation.id,
            claim="No deadline",
            confidence=50,
        )

        overdue = check_overdue_predictions(db)
        assert len(overdue) == 0

    def test_check_overdue_skips_already_resolved(
        self, db: Store, sample_simulation: Simulation,
    ) -> None:
        from forge.calibrate.resolver import check_overdue_predictions, resolve_prediction

        past = (datetime.now(UTC) - timedelta(days=30)).isoformat()
        p = db.save_prediction(
            simulation_id=sample_simulation.id,
            claim="Already resolved",
            confidence=60,
            resolution_deadline=past,
        )
        resolve_prediction(db, p.id, "true")

        overdue = check_overdue_predictions(db)
        assert len(overdue) == 0

    def test_store_list_predictions_pending(
        self, db: Store, sample_simulation: Simulation,
    ) -> None:
        p1 = db.save_prediction(
            simulation_id=sample_simulation.id, claim="Pending", confidence=50,
        )
        p2 = db.save_prediction(
            simulation_id=sample_simulation.id, claim="Resolved", confidence=50,
        )
        db.update_prediction(p2.id, resolved_as="true", resolved_at=datetime.now(UTC).isoformat())

        pending = db.list_predictions_pending()
        assert len(pending) == 1
        assert pending[0].id == p1.id

    def test_store_list_resolved_predictions(
        self, db: Store, sample_simulation: Simulation,
    ) -> None:
        p = db.save_prediction(
            simulation_id=sample_simulation.id, claim="Resolved", confidence=50,
        )
        db.update_prediction(p.id, resolved_as="false", resolved_at=datetime.now(UTC).isoformat())

        resolved = db.list_resolved_predictions()
        assert len(resolved) == 1
        assert resolved[0].resolved_as == "false"


# ------------------------------------------------------------------
# Calibration scoring
# ------------------------------------------------------------------


def _make_resolved_predictions(
    db: Store, simulation_id: str, data: list[tuple[int, str]],
) -> None:
    """Helper: create predictions with given (confidence, outcome) pairs."""
    now = datetime.now(UTC).isoformat()
    for confidence, outcome in data:
        p = db.save_prediction(
            simulation_id=simulation_id,
            claim=f"Prediction at {confidence}%",
            confidence=confidence,
        )
        db.update_prediction(p.id, resolved_as=outcome, resolved_at=now)


@pytest.mark.unit
class TestCalibrationScoring:
    def test_compute_brier_score_perfect(
        self, db: Store, sample_simulation: Simulation,
    ) -> None:
        from forge.calibrate.scorer import compute_brier_score

        # All 100% predictions resolved true = perfect score of 0
        _make_resolved_predictions(db, sample_simulation.id, [
            (100, "true"), (100, "true"), (0, "false"), (0, "false"),
        ])
        predictions = db.list_resolved_predictions()
        score = compute_brier_score(predictions)
        assert score == 0.0

    def test_compute_brier_score_worst(
        self, db: Store, sample_simulation: Simulation,
    ) -> None:
        from forge.calibrate.scorer import compute_brier_score

        # All 100% predictions resolved false = worst score of 1.0
        _make_resolved_predictions(db, sample_simulation.id, [
            (100, "false"), (0, "true"),
        ])
        predictions = db.list_resolved_predictions()
        score = compute_brier_score(predictions)
        assert score == 1.0

    def test_compute_brier_score_excludes_unresolvable(
        self, db: Store, sample_simulation: Simulation,
    ) -> None:
        from forge.calibrate.scorer import compute_brier_score

        _make_resolved_predictions(db, sample_simulation.id, [
            (80, "true"), (80, "unresolvable"),
        ])
        predictions = db.list_resolved_predictions()
        score = compute_brier_score(predictions)
        # Only one scorable prediction: (0.8 - 1)^2 = 0.04
        assert abs(score - 0.04) < 0.001

    def test_compute_brier_score_partial_counts_half(
        self, db: Store, sample_simulation: Simulation,
    ) -> None:
        from forge.calibrate.scorer import compute_brier_score

        _make_resolved_predictions(db, sample_simulation.id, [
            (50, "partial"),
        ])
        predictions = db.list_resolved_predictions()
        score = compute_brier_score(predictions)
        # (0.5 - 0.5)^2 = 0.0
        assert score == 0.0

    def test_compute_brier_score_empty_returns_zero(self) -> None:
        from forge.calibrate.scorer import compute_brier_score

        assert compute_brier_score([]) == 0.0

    def test_compute_calibration_buckets(
        self, db: Store, sample_simulation: Simulation,
    ) -> None:
        from forge.calibrate.scorer import compute_calibration

        _make_resolved_predictions(db, sample_simulation.id, [
            (75, "true"), (72, "true"), (78, "false"),  # 70-80 bucket
            (55, "true"), (52, "false"),                 # 50-60 bucket
        ])
        predictions = db.list_resolved_predictions()
        report = compute_calibration(predictions)

        assert report["total"] == 5
        assert report["resolved"] == 5
        buckets = {b["bucket"]: b for b in report["buckets"]}
        assert "70-80" in buckets
        assert buckets["70-80"]["total"] == 3
        assert buckets["70-80"]["correct"] == 2

    def test_compute_calibration_empty_predictions(self) -> None:
        from forge.calibrate.scorer import compute_calibration

        report = compute_calibration([])
        assert report["total"] == 0
        assert report["brier_score"] == 0.0

    def test_take_calibration_snapshot_saves(
        self, db: Store, sample_simulation: Simulation,
    ) -> None:
        from forge.calibrate.scorer import take_calibration_snapshot

        _make_resolved_predictions(db, sample_simulation.id, [
            (80, "true"), (60, "false"),
        ])

        snapshot = take_calibration_snapshot(db)
        assert snapshot.resolved_predictions == 2
        assert snapshot.accuracy_overall is not None

        # Verify it's persisted
        latest = db.get_latest_calibration_snapshot()
        assert latest is not None
        assert latest.id == snapshot.id

    def test_calibration_snapshot_stores_json(
        self, db: Store, sample_simulation: Simulation,
    ) -> None:
        import json

        from forge.calibrate.scorer import take_calibration_snapshot

        _make_resolved_predictions(db, sample_simulation.id, [
            (80, "true"), (80, "true"), (80, "false"),
        ])

        snapshot = take_calibration_snapshot(db)
        assert snapshot.calibration_json is not None
        data = json.loads(snapshot.calibration_json)
        assert isinstance(data, list)  # List of bucket dicts


# ------------------------------------------------------------------
# Drift detection
# ------------------------------------------------------------------


def _make_dated_predictions(
    db: Store,
    simulation_id: str,
    data: list[tuple[int, str, int]],
) -> None:
    """Helper: create predictions resolved N days ago.

    data: list of (confidence, outcome, days_ago).
    """
    for confidence, outcome, days_ago in data:
        p = db.save_prediction(
            simulation_id=simulation_id,
            claim=f"Prediction at {confidence}%",
            confidence=confidence,
        )
        resolved_at = (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()
        db.update_prediction(p.id, resolved_as=outcome, resolved_at=resolved_at)


@pytest.mark.unit
class TestDriftDetection:
    def test_detect_drift_no_data_returns_empty(self, db: Store) -> None:
        from forge.calibrate.drift import detect_drift

        alerts = detect_drift(db)
        assert alerts == []

    def test_detect_drift_stable_returns_empty(
        self, db: Store, sample_simulation: Simulation,
    ) -> None:
        from forge.calibrate.drift import detect_drift

        # Create good predictions in both windows
        _make_dated_predictions(db, sample_simulation.id, [
            (80, "true", 5), (80, "true", 10), (80, "true", 15),
            (80, "true", 35), (80, "true", 40), (80, "true", 45),
        ])

        alerts = detect_drift(db, window_days=30)
        assert alerts == []

    def test_detect_drift_brier_degradation(
        self, db: Store, sample_simulation: Simulation,
    ) -> None:
        from forge.calibrate.drift import detect_drift

        # Prior window: good predictions (Brier ~0)
        _make_dated_predictions(db, sample_simulation.id, [
            (90, "true", 40), (90, "true", 45), (90, "true", 50),
            (10, "false", 40), (10, "false", 45),
        ])
        # Current window: bad predictions (Brier ~1)
        _make_dated_predictions(db, sample_simulation.id, [
            (90, "false", 5), (90, "false", 10), (90, "false", 15),
            (10, "true", 5), (10, "true", 10),
        ])

        alerts = detect_drift(db, window_days=30)
        brier_alerts = [a for a in alerts if a.alert_type == "brier_degradation"]
        assert len(brier_alerts) >= 1

    def test_detect_drift_insufficient_data_no_alerts(
        self, db: Store, sample_simulation: Simulation,
    ) -> None:
        from forge.calibrate.drift import detect_drift

        # Only 1 prediction in each window — insufficient
        _make_dated_predictions(db, sample_simulation.id, [
            (80, "true", 5),
            (80, "true", 40),
        ])

        alerts = detect_drift(db, window_days=30)
        assert alerts == []
