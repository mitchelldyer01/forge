"""Repository pattern — all FORGE DB operations via raw SQL."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from ulid import ULID

from forge.db.models import Evidence, Feedback, Hypothesis, Relation
from forge.db.schema import apply_schema


def _now() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(UTC).isoformat()


def _gen_id(prefix: str) -> str:
    """Generate a ULID with the given type prefix."""
    return f"{prefix}{ULID()}"


class Store:
    """SQLite-backed store for all FORGE domain objects."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
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
