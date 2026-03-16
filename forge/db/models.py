"""Pydantic models for all FORGE domain objects."""

from __future__ import annotations

import json

from pydantic import BaseModel, field_validator


class Hypothesis(BaseModel):
    id: str
    claim: str
    context: str | None = None
    confidence: int = 50
    status: str = "alive"
    resolution_deadline: str | None = None
    generation: int = 0
    parent_id: str | None = None
    source: str
    source_ref: str | None = None
    created_at: str
    updated_at: str
    last_challenged_at: str | None = None
    challenges_survived: int = 0
    challenges_failed: int = 0
    human_endorsed: int = 0
    human_rejected: int = 0
    tags: list[str] | None = None
    embedding: bytes | None = None

    @field_validator("tags", mode="before")
    @classmethod
    def parse_tags(cls, v: str | list[str] | None) -> list[str] | None:
        if isinstance(v, str):
            return json.loads(v)
        return v


class Evidence(BaseModel):
    id: str
    content: str
    source_url: str | None = None
    source_name: str | None = None
    published_at: str | None = None
    ingested_at: str
    embedding: bytes | None = None


class Relation(BaseModel):
    id: str
    source_id: str
    target_id: str
    relation_type: str
    strength: float = 0.5
    reasoning: str | None = None
    created_at: str
    source_simulation_id: str | None = None


class Simulation(BaseModel):
    id: str
    mode: str
    seed_text: str
    seed_context: str | None = None
    agent_count: int | None = None
    rounds: int | None = None
    status: str = "pending"
    summary: str | None = None
    predictions_extracted: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None


class AgentPersona(BaseModel):
    id: str
    archetype: str
    persona_json: str
    simulations_participated: int = 0
    predictions_correct: int = 0
    predictions_incorrect: int = 0
    calibration_score: float | None = None
    active: int = 1
    created_at: str
    updated_at: str


class SimulationTurn(BaseModel):
    id: str
    simulation_id: str
    round: int
    agent_persona_id: str
    turn_type: str
    content: str
    responding_to_id: str | None = None
    position: str | None = None
    confidence: int | None = None
    token_count: int | None = None
    created_at: str


class Prediction(BaseModel):
    id: str
    simulation_id: str
    hypothesis_id: str | None = None
    claim: str
    confidence: int
    consensus_strength: float | None = None
    dissent_summary: str | None = None
    resolution_deadline: str | None = None
    resolved_at: str | None = None
    resolved_as: str | None = None
    resolution_evidence: str | None = None
    created_at: str


class Feed(BaseModel):
    id: str
    name: str
    url: str
    feed_type: str = "rss"
    active: int = 1
    last_polled_at: str | None = None
    poll_interval_minutes: int = 240


class Article(BaseModel):
    id: str
    feed_id: str | None = None
    url: str | None = None
    title: str | None = None
    content: str | None = None
    published_at: str | None = None
    ingested_at: str
    claims_extracted: int = 0


class Feedback(BaseModel):
    id: str
    hypothesis_id: str | None = None
    prediction_id: str | None = None
    action: str
    note: str | None = None
    created_at: str


class CalibrationSnapshot(BaseModel):
    id: str
    computed_at: str
    total_predictions: int | None = None
    resolved_predictions: int | None = None
    accuracy_overall: float | None = None
    calibration_json: str | None = None
    topic_breakdown_json: str | None = None
    archetype_breakdown_json: str | None = None
