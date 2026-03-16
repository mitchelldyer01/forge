"""Drift detection: alert when calibration degrades over time."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from forge.calibrate.scorer import compute_brier_score

if TYPE_CHECKING:
    from forge.db.models import Prediction
    from forge.db.store import Store

_MIN_PREDICTIONS = 3  # Minimum predictions per window to compute drift


@dataclass
class DriftAlert:
    alert_type: str  # "brier_degradation" | "topic_below_chance" | "archetype_degradation"
    description: str
    severity: str  # "warning" | "critical"
    metric_before: float
    metric_after: float


def detect_drift(
    store: Store,
    window_days: int = 30,
) -> list[DriftAlert]:
    """Detect calibration drift by comparing current vs prior window.

    Args:
        store: Database store.
        window_days: Size of each comparison window in days.

    Returns:
        List of drift alerts. Empty if no drift detected or insufficient data.
    """
    now = datetime.now(UTC)
    current_start = now - timedelta(days=window_days)
    prior_start = current_start - timedelta(days=window_days)

    resolved = store.list_resolved_predictions()
    if not resolved:
        return []

    # Split into windows by resolved_at
    current_window: list[Prediction] = []
    prior_window: list[Prediction] = []

    for p in resolved:
        if p.resolved_at is None:
            continue
        resolved_at = datetime.fromisoformat(p.resolved_at)
        if resolved_at >= current_start:
            current_window.append(p)
        elif resolved_at >= prior_start:
            prior_window.append(p)

    # Need minimum data in both windows
    if len(current_window) < _MIN_PREDICTIONS or len(prior_window) < _MIN_PREDICTIONS:
        return []

    alerts: list[DriftAlert] = []

    # Check Brier score degradation
    current_brier = compute_brier_score(current_window)
    prior_brier = compute_brier_score(prior_window)

    if prior_brier > 0:
        degradation = (current_brier - prior_brier) / prior_brier
        if degradation > 0.15:
            alerts.append(DriftAlert(
                alert_type="brier_degradation",
                description=(
                    f"Brier score degraded {degradation:.0%} "
                    f"({prior_brier:.3f} → {current_brier:.3f})"
                ),
                severity="critical" if degradation > 0.30 else "warning",
                metric_before=prior_brier,
                metric_after=current_brier,
            ))
    elif current_brier > 0.25:
        # Prior was perfect, now degraded
        alerts.append(DriftAlert(
            alert_type="brier_degradation",
            description=f"Brier score degraded from 0 to {current_brier:.3f}",
            severity="warning",
            metric_before=0.0,
            metric_after=current_brier,
        ))

    return alerts
