"""Calibration scoring: Brier score, bucket accuracy, snapshots."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.db.models import CalibrationSnapshot, Prediction
    from forge.db.store import Store

_OUTCOME_MAP = {"true": 1.0, "false": 0.0, "partial": 0.5}


def compute_brier_score(predictions: list[Prediction]) -> float:
    """Compute Brier score (proper scoring rule) for resolved predictions.

    Lower is better. 0.0 = perfect, 1.0 = worst possible.
    Excludes unresolvable predictions.
    """
    scorable = [
        p for p in predictions
        if p.resolved_as in _OUTCOME_MAP
    ]
    if not scorable:
        return 0.0

    total = 0.0
    for p in scorable:
        forecast = p.confidence / 100.0
        outcome = _OUTCOME_MAP[p.resolved_as]  # type: ignore[index]
        total += (forecast - outcome) ** 2

    return total / len(scorable)


def compute_calibration(predictions: list[Prediction]) -> dict:
    """Compute calibration report with bucketed accuracy.

    Returns dict with keys: total, resolved, accuracy, brier_score, buckets.
    """
    if not predictions:
        return {
            "total": 0,
            "resolved": 0,
            "accuracy": 0.0,
            "brier_score": 0.0,
            "buckets": [],
        }

    scorable = [p for p in predictions if p.resolved_as in _OUTCOME_MAP]
    correct = sum(1 for p in scorable if p.resolved_as == "true")
    accuracy = correct / len(scorable) if scorable else 0.0

    # Bucket by confidence decile
    bucket_data: dict[str, dict[str, int]] = {}
    for p in scorable:
        lower = (p.confidence // 10) * 10
        upper = lower + 10
        key = f"{lower}-{upper}"
        if key not in bucket_data:
            bucket_data[key] = {"total": 0, "correct": 0}
        bucket_data[key]["total"] += 1
        if p.resolved_as == "true":
            bucket_data[key]["correct"] += 1

    buckets = [
        {
            "bucket": k,
            "total": v["total"],
            "correct": v["correct"],
            "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0.0,
        }
        for k, v in sorted(bucket_data.items())
    ]

    return {
        "total": len(predictions),
        "resolved": len(scorable),
        "accuracy": accuracy,
        "brier_score": compute_brier_score(predictions),
        "buckets": buckets,
    }


def take_calibration_snapshot(store: Store) -> CalibrationSnapshot:
    """Compute calibration metrics and save a snapshot to the database."""
    resolved = store.list_resolved_predictions()
    report = compute_calibration(resolved)

    correct = sum(1 for p in resolved if p.resolved_as == "true")
    accuracy = correct / len(resolved) if resolved else None

    return store.save_calibration_snapshot(
        total_predictions=len(store.list_predictions()),
        resolved_predictions=len(resolved),
        accuracy_overall=accuracy,
        calibration_json=json.dumps(report["buckets"]),
    )
