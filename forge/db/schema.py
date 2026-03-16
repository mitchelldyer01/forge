"""SQLite schema definition and application for FORGE."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3

SCHEMA_SQL = """
-- Hypotheses: the core unit of knowledge
CREATE TABLE IF NOT EXISTS hypotheses (
    id TEXT PRIMARY KEY,
    claim TEXT NOT NULL,
    context TEXT,
    confidence INTEGER DEFAULT 50,
    status TEXT DEFAULT 'alive',
    resolution_deadline TEXT,
    generation INTEGER DEFAULT 0,
    parent_id TEXT,
    source TEXT NOT NULL,
    source_ref TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_challenged_at TEXT,
    challenges_survived INTEGER DEFAULT 0,
    challenges_failed INTEGER DEFAULT 0,
    human_endorsed INTEGER DEFAULT 0,
    human_rejected INTEGER DEFAULT 0,
    tags TEXT,
    embedding BLOB,
    FOREIGN KEY (parent_id) REFERENCES hypotheses(id)
);

-- Evidence: facts that support or contradict hypotheses
CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    source_url TEXT,
    source_name TEXT,
    published_at TEXT,
    ingested_at TEXT NOT NULL,
    embedding BLOB
);

-- Typed relationships
CREATE TABLE IF NOT EXISTS relations (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    strength REAL DEFAULT 0.5,
    reasoning TEXT,
    created_at TEXT NOT NULL,
    source_simulation_id TEXT,
    FOREIGN KEY (source_simulation_id) REFERENCES simulations(id)
);

-- Simulations: a complete simulation run
CREATE TABLE IF NOT EXISTS simulations (
    id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    seed_text TEXT NOT NULL,
    seed_context TEXT,
    agent_count INTEGER,
    rounds INTEGER,
    status TEXT DEFAULT 'pending',
    summary TEXT,
    predictions_extracted INTEGER DEFAULT 0,
    started_at TEXT,
    completed_at TEXT,
    duration_seconds REAL
);

-- Agent personas generated for simulations
CREATE TABLE IF NOT EXISTS agent_personas (
    id TEXT PRIMARY KEY,
    archetype TEXT NOT NULL,
    persona_json TEXT NOT NULL,
    simulations_participated INTEGER DEFAULT 0,
    predictions_correct INTEGER DEFAULT 0,
    predictions_incorrect INTEGER DEFAULT 0,
    calibration_score REAL,
    active INTEGER DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Individual agent reactions within a simulation
CREATE TABLE IF NOT EXISTS simulation_turns (
    id TEXT PRIMARY KEY,
    simulation_id TEXT NOT NULL,
    round INTEGER NOT NULL,
    agent_persona_id TEXT NOT NULL,
    turn_type TEXT NOT NULL,
    content TEXT NOT NULL,
    responding_to_id TEXT,
    position TEXT,
    confidence INTEGER,
    token_count INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (simulation_id) REFERENCES simulations(id),
    FOREIGN KEY (agent_persona_id) REFERENCES agent_personas(id)
);

-- Extracted predictions from simulations
CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    simulation_id TEXT NOT NULL,
    hypothesis_id TEXT,
    claim TEXT NOT NULL,
    confidence INTEGER NOT NULL,
    consensus_strength REAL,
    dissent_summary TEXT,
    resolution_deadline TEXT,
    resolved_at TEXT,
    resolved_as TEXT,
    resolution_evidence TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (simulation_id) REFERENCES simulations(id),
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(id)
);

-- RSS/feed sources
CREATE TABLE IF NOT EXISTS feeds (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    url TEXT NOT NULL UNIQUE,
    feed_type TEXT DEFAULT 'rss',
    active INTEGER DEFAULT 1,
    last_polled_at TEXT,
    poll_interval_minutes INTEGER DEFAULT 240
);

-- Ingested articles
CREATE TABLE IF NOT EXISTS articles (
    id TEXT PRIMARY KEY,
    feed_id TEXT,
    url TEXT UNIQUE,
    title TEXT,
    content TEXT,
    published_at TEXT,
    ingested_at TEXT NOT NULL,
    claims_extracted INTEGER DEFAULT 0,
    FOREIGN KEY (feed_id) REFERENCES feeds(id)
);

-- Human feedback
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    hypothesis_id TEXT,
    prediction_id TEXT,
    action TEXT NOT NULL,
    note TEXT,
    created_at TEXT NOT NULL
);

-- Calibration snapshots
CREATE TABLE IF NOT EXISTS calibration_snapshots (
    id TEXT PRIMARY KEY,
    computed_at TEXT NOT NULL,
    total_predictions INTEGER,
    resolved_predictions INTEGER,
    accuracy_overall REAL,
    calibration_json TEXT,
    topic_breakdown_json TEXT,
    archetype_breakdown_json TEXT
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_hypotheses_status ON hypotheses(status);
CREATE INDEX IF NOT EXISTS idx_hypotheses_confidence ON hypotheses(confidence);
CREATE INDEX IF NOT EXISTS idx_hypotheses_tags ON hypotheses(tags);
CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_id);
CREATE INDEX IF NOT EXISTS idx_simulations_status ON simulations(status);
CREATE INDEX IF NOT EXISTS idx_simulation_turns_sim ON simulation_turns(simulation_id);
CREATE INDEX IF NOT EXISTS idx_predictions_simulation ON predictions(simulation_id);
CREATE INDEX IF NOT EXISTS idx_predictions_resolved ON predictions(resolved_as);
CREATE INDEX IF NOT EXISTS idx_agent_personas_archetype ON agent_personas(archetype);
CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url);
"""


def apply_schema(conn: sqlite3.Connection) -> None:
    """Apply the full FORGE schema to a SQLite connection."""
    conn.executescript(SCHEMA_SQL)
