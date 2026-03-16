"""FastAPI server for FORGE."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

if TYPE_CHECKING:
    from forge.db.store import Store

VALID_FEEDBACK_ACTIONS = {
    "endorse", "reject", "resolve_true", "resolve_false",
    "resolve_partial", "annotate",
}


class FeedbackRequest(BaseModel):
    action: str
    hypothesis_id: str | None = None
    prediction_id: str | None = None
    note: str | None = None


def create_app(store: Store) -> FastAPI:
    """Create a FastAPI app wired to the given store."""
    app = FastAPI(title="FORGE API", version="0.1.0")

    @app.get("/v1/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/hypotheses")
    def list_hypotheses(
        status: str | None = None,
        min_confidence: int | None = None,
        max_confidence: int | None = None,
    ) -> list[dict]:
        hypotheses = store.list_hypotheses(
            status=status,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
        )
        return [h.model_dump(exclude={"embedding"}) for h in hypotheses]

    @app.get("/v1/hypotheses/{h_id}")
    def get_hypothesis(h_id: str) -> dict:
        h = store.get_hypothesis(h_id)
        if h is None:
            raise HTTPException(status_code=404, detail="Not found")
        relations = store.list_relations_for_hypothesis(h_id)
        data = h.model_dump(exclude={"embedding"})
        data["relations"] = [r.model_dump() for r in relations]
        return data

    @app.get("/v1/predictions")
    def list_predictions(
        resolved_as: str | None = None,
    ) -> list[dict]:
        predictions = store.list_predictions(resolved_as=resolved_as)
        return [p.model_dump() for p in predictions]

    @app.post("/v1/feedback")
    def post_feedback(req: FeedbackRequest) -> dict:
        if req.action not in VALID_FEEDBACK_ACTIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action. Must be one of: {VALID_FEEDBACK_ACTIONS}",
            )

        # Apply side effects for endorse/reject
        if req.action == "endorse" and req.hypothesis_id:
            h = store.get_hypothesis(req.hypothesis_id)
            if h:
                store.update_hypothesis(
                    h.id, human_endorsed=h.human_endorsed + 1,
                )
        elif req.action == "reject" and req.hypothesis_id:
            h = store.get_hypothesis(req.hypothesis_id)
            if h:
                store.update_hypothesis(
                    h.id, human_rejected=h.human_rejected + 1,
                )

        fb = store.save_feedback(
            action=req.action,
            hypothesis_id=req.hypothesis_id,
            prediction_id=req.prediction_id,
            note=req.note,
        )
        return {"id": fb.id, "action": fb.action}

    @app.get("/v1/calibration")
    def get_calibration() -> dict:
        from forge.calibrate.scorer import compute_calibration

        resolved = store.list_resolved_predictions()
        if not resolved:
            return {"total": 0, "resolved": 0, "accuracy": None, "brier_score": None, "buckets": []}
        report = compute_calibration(resolved)
        return report

    @app.get("/v1/leaderboard")
    def get_leaderboard() -> list[dict]:
        from forge.evolve.agent_evolution import get_leaderboard as _get_lb

        leaders = _get_lb(store, limit=10)
        return [a.model_dump() for a in leaders]

    @app.get("/v1/brief")
    def get_brief() -> dict:
        from forge.export.digest import generate_digest

        digest = generate_digest(store)
        return {
            "generated_at": digest.generated_at,
            "new_predictions": len(digest.new_predictions),
            "resolved_predictions": len(digest.resolved_predictions),
            "high_conviction": len(digest.high_conviction),
            "killed": len(digest.killed),
            "contradictions": len(digest.contradictions),
            "agent_leaderboard": len(digest.agent_leaderboard),
        }

    @app.get("/v1/stats")
    def get_stats() -> dict:
        h_counts = store.count_hypotheses_by_status()
        return {
            "total_hypotheses": sum(h_counts.values()),
            "hypotheses_by_status": h_counts,
            "total_evidence": store.count_evidence(),
            "total_feedback": store.count_feedback(),
        }

    @app.get("/v1/track-record", response_class=HTMLResponse)
    def track_record() -> str:
        from forge.calibrate.scorer import compute_calibration

        all_preds = store.list_predictions()
        resolved = store.list_resolved_predictions()
        report = compute_calibration(resolved) if resolved else None

        correct = [p for p in resolved if p.resolved_as == "true"]
        incorrect = [p for p in resolved if p.resolved_as == "false"]

        return _render_track_record(
            total=len(all_preds),
            resolved_count=len(resolved),
            correct=correct,
            incorrect=incorrect,
            report=report,
        )

    return app


def _render_track_record(
    total: int,
    resolved_count: int,
    correct: list,
    incorrect: list,
    report: dict | None,
) -> str:
    """Render track record as HTML."""
    accuracy = report["accuracy"] if report else 0
    brier = report["brier_score"] if report else None
    brier_str = f"{brier:.3f}" if brier is not None else "N/A"

    correct_html = ""
    for p in correct[:10]:
        correct_html += (
            f'<li class="correct">{_esc(p.claim)} '
            f"<span>({p.confidence}% confident)</span></li>\n"
        )

    incorrect_html = ""
    for p in incorrect[:10]:
        incorrect_html += (
            f'<li class="incorrect">{_esc(p.claim)} '
            f"<span>({p.confidence}% confident)</span></li>\n"
        )

    buckets_html = ""
    if report and report.get("buckets"):
        for b in report["buckets"]:
            buckets_html += (
                f"<tr><td>{b['bucket']}</td>"
                f"<td>{b['total']}</td>"
                f"<td>{b['correct']}</td>"
                f"<td>{b['accuracy']:.0%}</td></tr>\n"
            )

    no_preds = "No predictions yet." if total == 0 else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FORGE Track Record</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 2em auto; padding: 0 1em; }}
h1 {{ color: #1a1a2e; }}
.stats {{ display: flex; gap: 2em; margin: 1em 0; }}
.stat {{ text-align: center; }}
.stat .number {{ font-size: 2em; font-weight: bold; }}
.stat .label {{ color: #666; font-size: 0.9em; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
th {{ background: #f5f5f5; }}
.correct {{ color: #2d6a2d; }}
.incorrect {{ color: #a02020; }}
li span {{ color: #888; }}
</style>
</head>
<body>
<h1>FORGE Track Record</h1>
<p>{no_preds}</p>
<div class="stats">
<div class="stat"><div class="number">{total}</div><div class="label">Total Predictions</div></div>
<div class="stat"><div class="number">{resolved_count}</div><div class="label">Resolved</div></div>
<div class="stat"><div class="number">{accuracy:.0%}</div><div class="label">Accuracy</div></div>
<div class="stat"><div class="number">{brier_str}</div>\
<div class="label">Brier Score</div></div>
</div>

<h2>Calibration</h2>
{f'''<table>
<tr><th>Confidence</th><th>Total</th><th>Correct</th><th>Accuracy</th></tr>
{buckets_html}
</table>''' if buckets_html else '<p>Not enough data yet.</p>'}

<h2>Recent Correct Predictions</h2>
{f'<ul>{correct_html}</ul>' if correct_html else '<p>None yet.</p>'}

<h2>Recent Incorrect Predictions</h2>
{f'<ul>{incorrect_html}</ul>' if incorrect_html else '<p>None yet.</p>'}
</body>
</html>"""


def _esc(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
