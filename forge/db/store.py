"""Repository pattern — all FORGE DB operations via raw SQL."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from ulid import ULID

from forge.db.models import (
    AgentPersona,
    Article,
    CalibrationSnapshot,
    Evidence,
    Feed,
    Feedback,
    Hypothesis,
    Prediction,
    Relation,
    Simulation,
    SimulationTurn,
)
from forge.db.schema import apply_schema


def _now() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(UTC).isoformat()


def _gen_id(prefix: str) -> str:
    """Generate a ULID with the given type prefix."""
    return f"{prefix}{ULID()}"


class Store:
    """SQLite-backed store for all FORGE domain objects."""

    def __init__(self, db_path: str, *, check_same_thread: bool = True) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=check_same_thread)
        self.conn.row_factory = sqlite3.Row
        if db_path != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        apply_schema(self.conn)

    # ------------------------------------------------------------------
    # Hypothesis
    # ------------------------------------------------------------------

    def save_hypothesis(
        self,
        claim: str,
        source: str,
        *,
        context: str | None = None,
        confidence: int = 50,
        status: str = "alive",
        tags: list[str] | None = None,
        resolution_deadline: str | None = None,
        parent_id: str | None = None,
        source_ref: str | None = None,
    ) -> Hypothesis:
        if not claim:
            raise ValueError("claim must not be empty")
        if not source:
            raise ValueError("source must not be empty")

        now = _now()
        h_id = _gen_id("h_")
        tags_json = json.dumps(tags) if tags else None

        self.conn.execute(
            """INSERT INTO hypotheses
               (id, claim, context, confidence, status, resolution_deadline,
                generation, parent_id, source, source_ref, created_at, updated_at, tags)
               VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?)""",
            (h_id, claim, context, confidence, status, resolution_deadline,
             parent_id, source, source_ref, now, now, tags_json),
        )
        self.conn.commit()
        return self.get_hypothesis(h_id)  # type: ignore[return-value]

    def get_hypothesis(self, h_id: str) -> Hypothesis | None:
        row = self.conn.execute(
            "SELECT * FROM hypotheses WHERE id = ?", (h_id,)
        ).fetchone()
        if row is None:
            return None
        return Hypothesis(**dict(row))

    def update_hypothesis(self, h_id: str, **kwargs: object) -> Hypothesis | None:
        existing = self.get_hypothesis(h_id)
        if existing is None:
            return None

        kwargs["updated_at"] = _now()
        set_clause = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [h_id]
        self.conn.execute(
            f"UPDATE hypotheses SET {set_clause} WHERE id = ?",  # noqa: S608
            values,
        )
        self.conn.commit()
        return self.get_hypothesis(h_id)

    def list_hypotheses(
        self,
        *,
        status: str | None = None,
        min_confidence: int | None = None,
        max_confidence: int | None = None,
    ) -> list[Hypothesis]:
        query = "SELECT * FROM hypotheses WHERE 1=1"
        params: list[object] = []
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        if min_confidence is not None:
            query += " AND confidence >= ?"
            params.append(min_confidence)
        if max_confidence is not None:
            query += " AND confidence <= ?"
            params.append(max_confidence)
        query += " ORDER BY created_at DESC"

        rows = self.conn.execute(query, params).fetchall()
        return [Hypothesis(**dict(r)) for r in rows]

    # ------------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------------

    def save_evidence(
        self,
        content: str,
        *,
        source_url: str | None = None,
        source_name: str | None = None,
        published_at: str | None = None,
    ) -> Evidence:
        if not content:
            raise ValueError("content must not be empty")

        e_id = _gen_id("e_")
        now = _now()
        self.conn.execute(
            """INSERT INTO evidence
               (id, content, source_url, source_name, published_at, ingested_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (e_id, content, source_url, source_name, published_at, now),
        )
        self.conn.commit()
        return self.get_evidence(e_id)  # type: ignore[return-value]

    def get_evidence(self, e_id: str) -> Evidence | None:
        row = self.conn.execute(
            "SELECT * FROM evidence WHERE id = ?", (e_id,)
        ).fetchone()
        if row is None:
            return None
        return Evidence(**dict(row))

    # ------------------------------------------------------------------
    # Relation
    # ------------------------------------------------------------------

    def save_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        *,
        strength: float = 0.5,
        reasoning: str | None = None,
        source_simulation_id: str | None = None,
    ) -> Relation:
        r_id = _gen_id("r_")
        now = _now()
        self.conn.execute(
            """INSERT INTO relations
               (id, source_id, target_id, relation_type, strength, reasoning,
                created_at, source_simulation_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (r_id, source_id, target_id, relation_type, strength, reasoning,
             now, source_simulation_id),
        )
        self.conn.commit()
        return Relation(
            id=r_id, source_id=source_id, target_id=target_id,
            relation_type=relation_type, strength=strength, reasoning=reasoning,
            created_at=now, source_simulation_id=source_simulation_id,
        )

    def list_relations_by_source(self, source_id: str) -> list[Relation]:
        rows = self.conn.execute(
            "SELECT * FROM relations WHERE source_id = ?", (source_id,)
        ).fetchall()
        return [Relation(**dict(r)) for r in rows]

    def list_relations_for_hypothesis(self, h_id: str) -> list[Relation]:
        """List all relations where the hypothesis is source or target."""
        rows = self.conn.execute(
            "SELECT * FROM relations WHERE source_id = ? OR target_id = ?",
            (h_id, h_id),
        ).fetchall()
        return [Relation(**dict(r)) for r in rows]

    def list_relations_by_target(self, target_id: str) -> list[Relation]:
        rows = self.conn.execute(
            "SELECT * FROM relations WHERE target_id = ?", (target_id,)
        ).fetchall()
        return [Relation(**dict(r)) for r in rows]

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def save_feedback(
        self,
        action: str,
        *,
        hypothesis_id: str | None = None,
        prediction_id: str | None = None,
        note: str | None = None,
    ) -> Feedback:
        f_id = _gen_id("f_")
        now = _now()
        self.conn.execute(
            """INSERT INTO feedback (id, hypothesis_id, prediction_id, action, note, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (f_id, hypothesis_id, prediction_id, action, note, now),
        )
        self.conn.commit()
        return Feedback(
            id=f_id, hypothesis_id=hypothesis_id, prediction_id=prediction_id,
            action=action, note=note, created_at=now,
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def save_simulation(
        self,
        mode: str,
        seed_text: str,
        *,
        seed_context: str | None = None,
        agent_count: int | None = None,
        rounds: int | None = None,
    ) -> Simulation:
        if not seed_text:
            raise ValueError("seed_text must not be empty")

        s_id = _gen_id("s_")
        now = _now()
        self.conn.execute(
            """INSERT INTO simulations
               (id, mode, seed_text, seed_context, agent_count, rounds,
                status, started_at)
               VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)""",
            (s_id, mode, seed_text, seed_context, agent_count, rounds, now),
        )
        self.conn.commit()
        return self.get_simulation(s_id)  # type: ignore[return-value]

    def get_simulation(self, s_id: str) -> Simulation | None:
        row = self.conn.execute(
            "SELECT * FROM simulations WHERE id = ?", (s_id,)
        ).fetchone()
        if row is None:
            return None
        return Simulation(**dict(row))

    def update_simulation(self, s_id: str, **kwargs: object) -> Simulation | None:
        existing = self.get_simulation(s_id)
        if existing is None:
            return None
        set_clause = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [s_id]
        self.conn.execute(
            f"UPDATE simulations SET {set_clause} WHERE id = ?",  # noqa: S608
            values,
        )
        self.conn.commit()
        return self.get_simulation(s_id)

    def list_simulations(self, *, status: str | None = None) -> list[Simulation]:
        query = "SELECT * FROM simulations WHERE 1=1"
        params: list[object] = []
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY started_at DESC"
        rows = self.conn.execute(query, params).fetchall()
        return [Simulation(**dict(r)) for r in rows]

    # ------------------------------------------------------------------
    # AgentPersona
    # ------------------------------------------------------------------

    def save_agent_persona(
        self,
        archetype: str,
        persona_json: str,
    ) -> AgentPersona:
        ap_id = _gen_id("ap_")
        now = _now()
        self.conn.execute(
            """INSERT INTO agent_personas
               (id, archetype, persona_json, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (ap_id, archetype, persona_json, now, now),
        )
        self.conn.commit()
        return self.get_agent_persona(ap_id)  # type: ignore[return-value]

    def get_agent_persona(self, ap_id: str) -> AgentPersona | None:
        row = self.conn.execute(
            "SELECT * FROM agent_personas WHERE id = ?", (ap_id,)
        ).fetchone()
        if row is None:
            return None
        return AgentPersona(**dict(row))

    def update_agent_persona(self, ap_id: str, **kwargs: object) -> AgentPersona | None:
        existing = self.get_agent_persona(ap_id)
        if existing is None:
            return None
        kwargs["updated_at"] = _now()
        set_clause = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [ap_id]
        self.conn.execute(
            f"UPDATE agent_personas SET {set_clause} WHERE id = ?",  # noqa: S608
            values,
        )
        self.conn.commit()
        return self.get_agent_persona(ap_id)

    def list_agent_personas(
        self, *, active: bool | None = None,
    ) -> list[AgentPersona]:
        query = "SELECT * FROM agent_personas WHERE 1=1"
        params: list[object] = []
        if active is not None:
            query += " AND active = ?"
            params.append(1 if active else 0)
        query += " ORDER BY created_at DESC"
        rows = self.conn.execute(query, params).fetchall()
        return [AgentPersona(**dict(r)) for r in rows]

    # ------------------------------------------------------------------
    # SimulationTurn
    # ------------------------------------------------------------------

    def save_simulation_turn(
        self,
        simulation_id: str,
        round: int,
        agent_persona_id: str,
        turn_type: str,
        content: str,
        *,
        responding_to_id: str | None = None,
        position: str | None = None,
        confidence: int | None = None,
        token_count: int | None = None,
        raw_content: str | None = None,
    ) -> SimulationTurn:
        if not content:
            raise ValueError("content must not be empty")

        st_id = _gen_id("st_")
        now = _now()
        self.conn.execute(
            """INSERT INTO simulation_turns
               (id, simulation_id, round, agent_persona_id, turn_type, content,
                responding_to_id, position, confidence, token_count,
                raw_content, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (st_id, simulation_id, round, agent_persona_id, turn_type, content,
             responding_to_id, position, confidence, token_count,
             raw_content, now),
        )
        self.conn.commit()
        return SimulationTurn(
            id=st_id, simulation_id=simulation_id, round=round,
            agent_persona_id=agent_persona_id, turn_type=turn_type,
            content=content, responding_to_id=responding_to_id,
            position=position, confidence=confidence,
            token_count=token_count, raw_content=raw_content,
            created_at=now,
        )

    def list_turns_by_simulation(
        self, simulation_id: str, *, round: int | None = None,
    ) -> list[SimulationTurn]:
        query = "SELECT * FROM simulation_turns WHERE simulation_id = ?"
        params: list[object] = [simulation_id]
        if round is not None:
            query += " AND round = ?"
            params.append(round)
        query += " ORDER BY created_at"
        rows = self.conn.execute(query, params).fetchall()
        return [SimulationTurn(**dict(r)) for r in rows]

    def list_turns_by_agent(
        self, simulation_id: str, agent_persona_id: str,
    ) -> list[SimulationTurn]:
        rows = self.conn.execute(
            """SELECT * FROM simulation_turns
               WHERE simulation_id = ? AND agent_persona_id = ?
               ORDER BY created_at""",
            (simulation_id, agent_persona_id),
        ).fetchall()
        return [SimulationTurn(**dict(r)) for r in rows]

    def list_turns_with_agent(
        self,
        simulation_id: str,
        *,
        round: int | None = None,
        archetype: str | None = None,
    ) -> list[dict]:
        """List turns joined with agent persona info.

        Returns list of dicts with turn fields + archetype + persona_json.
        """
        query = """
            SELECT st.*, ap.archetype, ap.persona_json
            FROM simulation_turns st
            JOIN agent_personas ap ON st.agent_persona_id = ap.id
            WHERE st.simulation_id = ?
        """
        params: list[object] = [simulation_id]
        if round is not None:
            query += " AND st.round = ?"
            params.append(round)
        if archetype is not None:
            query += " AND ap.archetype LIKE ?"
            params.append(f"%{archetype}%")
        query += " ORDER BY st.round, st.created_at"
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def save_prediction(
        self,
        simulation_id: str,
        claim: str,
        confidence: int,
        *,
        consensus_strength: float | None = None,
        dissent_summary: str | None = None,
        resolution_deadline: str | None = None,
    ) -> Prediction:
        if not claim:
            raise ValueError("claim must not be empty")

        p_id = _gen_id("p_")
        now = _now()
        self.conn.execute(
            """INSERT INTO predictions
               (id, simulation_id, claim, confidence, consensus_strength,
                dissent_summary, resolution_deadline, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (p_id, simulation_id, claim, confidence, consensus_strength,
             dissent_summary, resolution_deadline, now),
        )
        self.conn.commit()
        return self.get_prediction(p_id)  # type: ignore[return-value]

    def get_prediction(self, p_id: str) -> Prediction | None:
        row = self.conn.execute(
            "SELECT * FROM predictions WHERE id = ?", (p_id,)
        ).fetchone()
        if row is None:
            return None
        return Prediction(**dict(row))

    def update_prediction(self, p_id: str, **kwargs: object) -> Prediction | None:
        existing = self.get_prediction(p_id)
        if existing is None:
            return None
        set_clause = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [p_id]
        self.conn.execute(
            f"UPDATE predictions SET {set_clause} WHERE id = ?",  # noqa: S608
            values,
        )
        self.conn.commit()
        return self.get_prediction(p_id)

    def list_predictions(
        self, *, simulation_id: str | None = None, resolved_as: str | None = None,
    ) -> list[Prediction]:
        query = "SELECT * FROM predictions WHERE 1=1"
        params: list[object] = []
        if simulation_id is not None:
            query += " AND simulation_id = ?"
            params.append(simulation_id)
        if resolved_as is not None:
            query += " AND resolved_as = ?"
            params.append(resolved_as)
        query += " ORDER BY created_at DESC"
        rows = self.conn.execute(query, params).fetchall()
        return [Prediction(**dict(r)) for r in rows]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def count_hypotheses_by_status(self) -> dict[str, int]:
        """Count hypotheses grouped by status."""
        rows = self.conn.execute(
            "SELECT status, COUNT(*) as cnt FROM hypotheses GROUP BY status"
        ).fetchall()
        return {row["status"]: row["cnt"] for row in rows}

    def count_evidence(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM evidence").fetchone()
        return row["cnt"]

    def count_feedback(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM feedback").fetchone()
        return row["cnt"]

    # ------------------------------------------------------------------
    # Feed
    # ------------------------------------------------------------------

    def save_feed(
        self,
        name: str,
        url: str,
        *,
        feed_type: str = "rss",
        poll_interval_minutes: int = 240,
    ) -> Feed:
        if not name:
            raise ValueError("name must not be empty")
        if not url:
            raise ValueError("url must not be empty")

        f_id = _gen_id("f_")
        self.conn.execute(
            """INSERT INTO feeds
               (id, name, url, feed_type, active, poll_interval_minutes)
               VALUES (?, ?, ?, ?, 1, ?)""",
            (f_id, name, url, feed_type, poll_interval_minutes),
        )
        self.conn.commit()
        return self.get_feed(f_id)  # type: ignore[return-value]

    def get_feed(self, feed_id: str) -> Feed | None:
        row = self.conn.execute(
            "SELECT * FROM feeds WHERE id = ?", (feed_id,)
        ).fetchone()
        if row is None:
            return None
        return Feed(**dict(row))

    def list_feeds(self, *, active: bool | None = None) -> list[Feed]:
        query = "SELECT * FROM feeds WHERE 1=1"
        params: list[object] = []
        if active is not None:
            query += " AND active = ?"
            params.append(1 if active else 0)
        rows = self.conn.execute(query, params).fetchall()
        return [Feed(**dict(r)) for r in rows]

    def update_feed(self, feed_id: str, **kwargs: object) -> Feed | None:
        existing = self.get_feed(feed_id)
        if existing is None:
            return None
        set_clause = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [feed_id]
        self.conn.execute(
            f"UPDATE feeds SET {set_clause} WHERE id = ?",  # noqa: S608
            values,
        )
        self.conn.commit()
        return self.get_feed(feed_id)

    # ------------------------------------------------------------------
    # Article
    # ------------------------------------------------------------------

    def save_article(
        self,
        url: str | None = None,
        *,
        feed_id: str | None = None,
        title: str | None = None,
        content: str | None = None,
        published_at: str | None = None,
    ) -> Article:
        a_id = _gen_id("a_")
        now = _now()
        self.conn.execute(
            """INSERT INTO articles
               (id, feed_id, url, title, content, published_at, ingested_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (a_id, feed_id, url, title, content, published_at, now),
        )
        self.conn.commit()
        return self.get_article(a_id)  # type: ignore[return-value]

    def get_article(self, article_id: str) -> Article | None:
        row = self.conn.execute(
            "SELECT * FROM articles WHERE id = ?", (article_id,)
        ).fetchone()
        if row is None:
            return None
        return Article(**dict(row))

    def get_article_by_url(self, url: str) -> Article | None:
        row = self.conn.execute(
            "SELECT * FROM articles WHERE url = ?", (url,)
        ).fetchone()
        if row is None:
            return None
        return Article(**dict(row))

    def list_articles(
        self,
        *,
        feed_id: str | None = None,
        unextracted: bool = False,
    ) -> list[Article]:
        query = "SELECT * FROM articles WHERE 1=1"
        params: list[object] = []
        if feed_id is not None:
            query += " AND feed_id = ?"
            params.append(feed_id)
        if unextracted:
            query += " AND claims_extracted = 0"
        query += " ORDER BY ingested_at DESC"
        rows = self.conn.execute(query, params).fetchall()
        return [Article(**dict(r)) for r in rows]

    def update_article(self, article_id: str, **kwargs: object) -> Article | None:
        existing = self.get_article(article_id)
        if existing is None:
            return None
        set_clause = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [article_id]
        self.conn.execute(
            f"UPDATE articles SET {set_clause} WHERE id = ?",  # noqa: S608
            values,
        )
        self.conn.commit()
        return self.get_article(article_id)

    # ------------------------------------------------------------------
    # CalibrationSnapshot
    # ------------------------------------------------------------------

    def save_calibration_snapshot(
        self,
        *,
        total_predictions: int = 0,
        resolved_predictions: int = 0,
        accuracy_overall: float | None = None,
        calibration_json: str | None = None,
        topic_breakdown_json: str | None = None,
        archetype_breakdown_json: str | None = None,
    ) -> CalibrationSnapshot:
        cs_id = _gen_id("cs_")
        now = _now()
        self.conn.execute(
            """INSERT INTO calibration_snapshots
               (id, computed_at, total_predictions, resolved_predictions,
                accuracy_overall, calibration_json, topic_breakdown_json,
                archetype_breakdown_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (cs_id, now, total_predictions, resolved_predictions,
             accuracy_overall, calibration_json, topic_breakdown_json,
             archetype_breakdown_json),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT * FROM calibration_snapshots WHERE id = ?", (cs_id,)
        ).fetchone()
        return CalibrationSnapshot(**dict(row))

    def get_latest_calibration_snapshot(self) -> CalibrationSnapshot | None:
        row = self.conn.execute(
            "SELECT * FROM calibration_snapshots ORDER BY computed_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return CalibrationSnapshot(**dict(row))

    def list_calibration_snapshots(self) -> list[CalibrationSnapshot]:
        rows = self.conn.execute(
            "SELECT * FROM calibration_snapshots ORDER BY computed_at DESC"
        ).fetchall()
        return [CalibrationSnapshot(**dict(r)) for r in rows]

    # ------------------------------------------------------------------
    # Prediction queries (Phase 3 additions)
    # ------------------------------------------------------------------

    def list_predictions_past_deadline(self) -> list[Prediction]:
        """Predictions past deadline that haven't been resolved."""
        now = _now()
        rows = self.conn.execute(
            """SELECT * FROM predictions
               WHERE resolution_deadline IS NOT NULL
               AND resolution_deadline < ?
               AND resolved_as IS NULL
               ORDER BY resolution_deadline""",
            (now,),
        ).fetchall()
        return [Prediction(**dict(r)) for r in rows]

    def list_predictions_pending(self) -> list[Prediction]:
        """Predictions that haven't been resolved yet."""
        rows = self.conn.execute(
            """SELECT * FROM predictions
               WHERE resolved_as IS NULL
               ORDER BY created_at DESC"""
        ).fetchall()
        return [Prediction(**dict(r)) for r in rows]

    def list_resolved_predictions(self) -> list[Prediction]:
        """Predictions that have been resolved."""
        rows = self.conn.execute(
            """SELECT * FROM predictions
               WHERE resolved_as IS NOT NULL
               ORDER BY resolved_at DESC"""
        ).fetchall()
        return [Prediction(**dict(r)) for r in rows]
