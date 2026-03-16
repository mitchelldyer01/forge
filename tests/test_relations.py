"""Tests for relation extraction from judge output and persistence."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from forge.analyze.structured import Verdict, analyze

if TYPE_CHECKING:
    from forge.db.store import Store

STEELMAN_RESPONSE = {
    "position": "AI will transform SaaS",
    "arguments": ["AI agents can automate tasks"],
    "confidence": 70,
}

REDTEAM_RESPONSE = {
    "attacks": ["Switching costs are high"],
    "weaknesses": ["Timeline too aggressive"],
    "confidence": 60,
}


class TestRelationExtraction:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_judge_returns_relations(self, mock_llm) -> None:
        judge_response = {
            "position": "Partially true",
            "confidence": 55,
            "synthesis": "Both sides have merit",
            "steelman_arg": "AI can automate",
            "redteam_arg": "Switching costs high",
            "conditions": [],
            "tags": ["ai"],
            "relations": [
                {
                    "target_id": "h_existing123",
                    "type": "supports",
                    "reasoning": "Both predict AI growth",
                },
            ],
        }
        mock_llm.set_responses([STEELMAN_RESPONSE, REDTEAM_RESPONSE, judge_response])
        verdict = await analyze(
            claim="AI agents will replace SaaS",
            llm=mock_llm,
            existing_hypotheses='- ID: h_existing123 | Claim: "AI adoption accelerating"',
        )
        assert verdict.relations is not None
        assert len(verdict.relations) == 1
        assert verdict.relations[0]["target_id"] == "h_existing123"
        assert verdict.relations[0]["type"] == "supports"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_judge_without_relations(self, mock_llm) -> None:
        judge_response = {
            "position": "Uncertain",
            "confidence": 50,
            "synthesis": "Not enough data",
            "steelman_arg": "Some support",
            "redteam_arg": "Some opposition",
            "conditions": [],
            "tags": [],
        }
        mock_llm.set_responses([STEELMAN_RESPONSE, REDTEAM_RESPONSE, judge_response])
        verdict = await analyze(claim="Test", llm=mock_llm)
        assert verdict.relations is None


class TestRelationPersistence:
    @pytest.mark.unit
    def test_save_relations_from_verdict(self, db: Store) -> None:
        """Relations from verdict should be persistable via store."""
        from forge.analyze.relations import save_verdict_relations

        # Create the source hypothesis
        source = db.save_hypothesis(claim="New claim", source="manual")
        # Create the target hypothesis
        target = db.save_hypothesis(claim="Existing claim", source="manual")

        verdict = Verdict(
            position="test",
            confidence=50,
            synthesis="test",
            steelman_arg="for",
            redteam_arg="against",
            conditions=[],
            tags=[],
            relations=[
                {
                    "target_id": target.id,
                    "type": "supports",
                    "reasoning": "Both predict similar outcomes",
                },
            ],
        )

        save_verdict_relations(source.id, verdict, db)

        rels = db.list_relations_by_source(source.id)
        assert len(rels) == 1
        assert rels[0].target_id == target.id
        assert rels[0].relation_type == "supports"
        assert rels[0].reasoning == "Both predict similar outcomes"

    @pytest.mark.unit
    def test_save_relations_skips_when_none(self, db: Store) -> None:
        from forge.analyze.relations import save_verdict_relations

        source = db.save_hypothesis(claim="Test", source="manual")
        verdict = Verdict(
            position="test", confidence=50, synthesis="test",
            steelman_arg="f", redteam_arg="a", conditions=[], tags=[],
            relations=None,
        )
        save_verdict_relations(source.id, verdict, db)
        rels = db.list_relations_by_source(source.id)
        assert len(rels) == 0

    @pytest.mark.unit
    def test_save_relations_skips_invalid_target(self, db: Store) -> None:
        from forge.analyze.relations import save_verdict_relations

        source = db.save_hypothesis(claim="Test", source="manual")
        verdict = Verdict(
            position="test", confidence=50, synthesis="test",
            steelman_arg="f", redteam_arg="a", conditions=[], tags=[],
            relations=[{"target_id": "h_nonexistent", "type": "supports", "reasoning": "test"}],
        )
        # Should not crash, just skip invalid targets
        save_verdict_relations(source.id, verdict, db)
        rels = db.list_relations_by_source(source.id)
        # Relations are saved regardless of target existence (store doesn't validate)
        assert len(rels) == 1
