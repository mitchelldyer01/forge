"""
Tests for persistence wiring — pipeline output saved to DB.

Verifies that running structured analysis persists hypothesis to Store.
"""

import pytest

from forge.analyze.structured import Verdict
from forge.db.models import Hypothesis
from forge.db.store import Store

# ---------------------------------------------------------------------------
# save_verdict_as_hypothesis
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPersistence:
    def test_save_verdict_creates_hypothesis(self, db: Store):
        """Saving a verdict creates a hypothesis in the DB."""
        from forge.analyze.persistence import save_verdict

        verdict = Verdict(
            position="support",
            confidence=72,
            synthesis="The claim is likely true.",
            steelman_arg="Strong case",
            redteam_arg="Weak counter",
            conditions=["stable market"],
            tags=["economics"],
        )

        h = save_verdict(db, claim="Test claim", verdict=verdict)
        assert isinstance(h, Hypothesis)
        assert h.id.startswith("h_")
        assert h.claim == "Test claim"
        assert h.confidence == 72
        assert h.source == "manual"
        assert h.tags == ["economics"]

    def test_save_verdict_with_context(self, db: Store):
        """Saved hypothesis includes context."""
        from forge.analyze.persistence import save_verdict

        verdict = Verdict(
            position="oppose",
            confidence=30,
            synthesis="Unlikely.",
            steelman_arg="for",
            redteam_arg="against",
        )

        h = save_verdict(db, claim="Test", verdict=verdict, context="Background info")
        assert h.context == "Background info"

    def test_save_verdict_persists_to_db(self, db: Store):
        """Hypothesis is retrievable from DB after save."""
        from forge.analyze.persistence import save_verdict

        verdict = Verdict(
            position="conditional",
            confidence=55,
            synthesis="Maybe.",
            steelman_arg="for",
            redteam_arg="against",
            tags=["ai"],
        )

        h = save_verdict(db, claim="Persisted claim", verdict=verdict)
        found = db.get_hypothesis(h.id)
        assert found is not None
        assert found.claim == "Persisted claim"
        assert found.confidence == 55

    def test_save_multiple_verdicts(self, db: Store):
        """Multiple verdicts create multiple hypotheses."""
        from forge.analyze.persistence import save_verdict

        for i in range(3):
            verdict = Verdict(
                position="support",
                confidence=60 + i * 10,
                synthesis=f"Verdict {i}",
                steelman_arg="for",
                redteam_arg="against",
            )
            save_verdict(db, claim=f"Claim {i}", verdict=verdict)

        all_h = db.list_hypotheses()
        assert len(all_h) == 3
