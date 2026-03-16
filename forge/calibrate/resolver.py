"""Resolution tracking for predictions."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.db.models import Prediction
    from forge.db.store import Store

_VALID_OUTCOMES = {"true", "false", "partial", "unresolvable"}


def resolve_prediction(
    store: Store,
    prediction_id: str,
    outcome: str,
    *,
    evidence: str | None = None,
    note: str | None = None,
) -> Prediction:
    """Resolve a prediction with an outcome.

    Args:
        store: Database store.
        prediction_id: The prediction ID (p_...).
        outcome: One of "true", "false", "partial", "unresolvable".
        evidence: Optional resolution evidence text.
        note: Optional note for feedback record.

    Returns:
        Updated Prediction model.

    Raises:
        ValueError: If outcome is invalid or prediction not found.
    """
    if outcome not in _VALID_OUTCOMES:
        raise ValueError(f"Invalid outcome '{outcome}'. Must be one of: {_VALID_OUTCOMES}")

    existing = store.get_prediction(prediction_id)
    if existing is None:
        raise ValueError(f"Prediction {prediction_id} not found")

    now = datetime.now(UTC).isoformat()
    updated = store.update_prediction(
        prediction_id,
        resolved_as=outcome,
        resolved_at=now,
        resolution_evidence=evidence,
    )

    # Save feedback record
    store.save_feedback(
        action=f"resolve_{outcome}",
        prediction_id=prediction_id,
        note=note,
    )

    return updated  # type: ignore[return-value]


def check_overdue_predictions(store: Store) -> list[Prediction]:
    """Return predictions past their resolution deadline that are unresolved."""
    return store.list_predictions_past_deadline()
