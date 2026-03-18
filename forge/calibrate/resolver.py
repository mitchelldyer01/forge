"""Resolution tracking for predictions."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.db.models import Prediction
    from forge.db.store import Store

logger = logging.getLogger(__name__)

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

    # Update agent calibration scores
    if outcome in ("true", "false", "partial"):
        agents_updated = update_agent_scores(store, prediction_id, outcome)
        if agents_updated > 0:
            logger.info(
                "Updated %d agent scores after resolving %s as %s",
                agents_updated, prediction_id, outcome,
            )

    return updated  # type: ignore[return-value]


def update_agent_scores(
    store: Store,
    prediction_id: str,
    outcome: str,
) -> int:
    """Trace prediction → simulation → agents and update calibration scores.

    For each agent who participated in Round 3 of the prediction's simulation,
    determines whether their final position aligned with the outcome and
    updates their predictions_correct/predictions_incorrect/calibration_score.

    Args:
        store: Database store.
        prediction_id: The resolved prediction ID.
        outcome: The resolution outcome ("true", "false", "partial").

    Returns:
        Number of agents updated.
    """
    prediction = store.get_prediction(prediction_id)
    if prediction is None or not prediction.simulation_id:
        return 0

    turns = store.list_turns_by_simulation(prediction.simulation_id, round=3)
    if not turns:
        return 0

    updated = 0
    seen_agents: set[str] = set()
    for turn in turns:
        if turn.agent_persona_id in seen_agents:
            continue
        seen_agents.add(turn.agent_persona_id)

        agent = store.get_agent_persona(turn.agent_persona_id)
        if agent is None:
            continue

        position = (turn.position or "").lower()
        correct = _agent_was_correct(position, outcome)

        new_correct = agent.predictions_correct + (1 if correct else 0)
        new_incorrect = agent.predictions_incorrect + (0 if correct else 1)
        total = new_correct + new_incorrect
        score = new_correct / total if total > 0 else None

        store.update_agent_persona(
            agent.id,
            predictions_correct=new_correct,
            predictions_incorrect=new_incorrect,
            calibration_score=score,
        )
        updated += 1

    logger.info(
        "Updated %d agent scores for prediction %s (outcome=%s)",
        updated, prediction_id, outcome,
    )
    return updated


def _agent_was_correct(position: str, outcome: str) -> bool:
    """Determine if an agent's final position aligned with the outcome."""
    if outcome == "partial":
        return True
    supportive = position in ("support", "strongly_support", "agree")
    outcome_true = outcome == "true"
    return supportive == outcome_true


def check_overdue_predictions(store: Store) -> list[Prediction]:
    """Return predictions past their resolution deadline that are unresolved."""
    return store.list_predictions_past_deadline()
