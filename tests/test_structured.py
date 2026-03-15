"""
Tests for forge.analyze.structured — steelman -> redteam -> judge pipeline.

Mirrors: forge/analyze/structured.py
"""

import pytest

from forge.analyze.structured import Verdict, run_structured_analysis

# ---------------------------------------------------------------------------
# Verdict model
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestVerdictModel:
    def test_verdict_has_required_fields(self):
        """Verdict model has position, confidence, and synthesis."""
        v = Verdict(
            position="support",
            confidence=75,
            synthesis="The claim is likely true.",
            steelman_arg="Strong case",
            redteam_arg="Weak attack",
            conditions=["if market stays stable"],
            tags=["economics"],
        )
        assert v.position == "support"
        assert v.confidence == 75
        assert v.synthesis == "The claim is likely true."

    def test_verdict_stores_steelman_and_redteam(self):
        """Verdict preserves both sides of the analysis."""
        v = Verdict(
            position="oppose",
            confidence=30,
            synthesis="Unlikely.",
            steelman_arg="Best case for it",
            redteam_arg="Devastating counter",
        )
        assert v.steelman_arg == "Best case for it"
        assert v.redteam_arg == "Devastating counter"

    def test_verdict_optional_fields_default_none(self):
        """Conditions and tags default to None when not provided."""
        v = Verdict(
            position="conditional",
            confidence=50,
            synthesis="Uncertain.",
            steelman_arg="case for",
            redteam_arg="case against",
        )
        assert v.conditions is None
        assert v.tags is None


# ---------------------------------------------------------------------------
# Full pipeline with MockLLMClient
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestStructuredAnalysisPipeline:
    @pytest.mark.asyncio
    async def test_pipeline_returns_verdict(self, mock_llm):
        """run_structured_analysis returns a Verdict model."""
        mock_llm.set_responses([
            {
                "argument": "Strong case for the claim",
                "key_evidence": ["evidence 1"],
                "conditions": ["condition 1"],
                "confidence_if_true": 80,
            },
            {
                "attack": "Strong case against",
                "counterevidence": ["counter 1"],
                "logical_flaws": ["flaw 1"],
                "alternative_explanations": ["alt 1"],
                "confidence_if_false": 60,
            },
            {
                "position": "support",
                "confidence": 72,
                "synthesis": "On balance, the claim holds.",
                "steelman_strength": "Compelling evidence",
                "redteam_strength": "Minor flaws found",
                "conditions": ["stable market"],
                "tags": ["economics", "prediction"],
            },
        ])

        verdict = await run_structured_analysis(
            mock_llm, claim="Test claim", context="Some context"
        )

        assert isinstance(verdict, Verdict)
        assert verdict.position == "support"
        assert verdict.confidence == 72

    @pytest.mark.asyncio
    async def test_pipeline_makes_three_llm_calls(self, mock_llm):
        """Pipeline makes exactly 3 LLM calls: steelman, redteam, judge."""
        mock_llm.set_responses([
            {"argument": "for", "key_evidence": [], "conditions": [], "confidence_if_true": 70},
            {"attack": "against", "counterevidence": [], "logical_flaws": [],
             "alternative_explanations": [], "confidence_if_false": 50},
            {"position": "support", "confidence": 65, "synthesis": "Likely true.",
             "steelman_strength": "ok", "redteam_strength": "ok",
             "conditions": [], "tags": []},
        ])

        await run_structured_analysis(mock_llm, claim="Test")
        assert mock_llm.call_count == 3

    @pytest.mark.asyncio
    async def test_pipeline_passes_steelman_to_redteam(self, mock_llm):
        """Steelman output is passed as context to redteam call."""
        call_messages = []
        original_complete = mock_llm.complete

        async def tracking_complete(messages, **kwargs):
            call_messages.append(messages)
            return await original_complete(messages, **kwargs)

        mock_llm.complete = tracking_complete

        mock_llm.set_responses([
            {"argument": "Strong steelman argument here", "key_evidence": [],
             "conditions": [], "confidence_if_true": 80},
            {"attack": "counter", "counterevidence": [], "logical_flaws": [],
             "alternative_explanations": [], "confidence_if_false": 40},
            {"position": "support", "confidence": 70, "synthesis": "ok",
             "steelman_strength": "ok", "redteam_strength": "ok",
             "conditions": [], "tags": []},
        ])

        await run_structured_analysis(mock_llm, claim="Test claim")

        # Second call (redteam) should reference steelman output
        redteam_text = " ".join(m["content"] for m in call_messages[1])
        assert "Strong steelman argument here" in redteam_text

    @pytest.mark.asyncio
    async def test_pipeline_passes_both_to_judge(self, mock_llm):
        """Judge receives both steelman and redteam arguments."""
        call_messages = []
        original_complete = mock_llm.complete

        async def tracking_complete(messages, **kwargs):
            call_messages.append(messages)
            return await original_complete(messages, **kwargs)

        mock_llm.complete = tracking_complete

        mock_llm.set_responses([
            {"argument": "STEELMAN_OUTPUT", "key_evidence": [],
             "conditions": [], "confidence_if_true": 80},
            {"attack": "REDTEAM_OUTPUT", "counterevidence": [], "logical_flaws": [],
             "alternative_explanations": [], "confidence_if_false": 40},
            {"position": "oppose", "confidence": 35, "synthesis": "Unlikely.",
             "steelman_strength": "ok", "redteam_strength": "ok",
             "conditions": [], "tags": []},
        ])

        await run_structured_analysis(mock_llm, claim="Test claim")

        # Third call (judge) should reference both outputs
        judge_text = " ".join(m["content"] for m in call_messages[2])
        assert "STEELMAN_OUTPUT" in judge_text
        assert "REDTEAM_OUTPUT" in judge_text

    @pytest.mark.asyncio
    async def test_pipeline_preserves_verdict_tags(self, mock_llm):
        """Tags from judge output are preserved in Verdict."""
        mock_llm.set_responses([
            {"argument": "for", "key_evidence": [], "conditions": [],
             "confidence_if_true": 70},
            {"attack": "against", "counterevidence": [], "logical_flaws": [],
             "alternative_explanations": [], "confidence_if_false": 50},
            {"position": "support", "confidence": 65, "synthesis": "ok",
             "steelman_strength": "ok", "redteam_strength": "ok",
             "conditions": ["if stable"], "tags": ["ai", "pricing"]},
        ])

        verdict = await run_structured_analysis(mock_llm, claim="Test")
        assert verdict.tags == ["ai", "pricing"]
        assert verdict.conditions == ["if stable"]

    @pytest.mark.asyncio
    async def test_pipeline_without_context(self, mock_llm):
        """Pipeline works without context parameter."""
        mock_llm.set_responses([
            {"argument": "for", "key_evidence": [], "conditions": [],
             "confidence_if_true": 70},
            {"attack": "against", "counterevidence": [], "logical_flaws": [],
             "alternative_explanations": [], "confidence_if_false": 50},
            {"position": "conditional", "confidence": 55, "synthesis": "unclear",
             "steelman_strength": "ok", "redteam_strength": "ok",
             "conditions": [], "tags": []},
        ])

        verdict = await run_structured_analysis(mock_llm, claim="Test")
        assert isinstance(verdict, Verdict)
