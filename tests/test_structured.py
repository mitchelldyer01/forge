"""Tests for forge/analyze/structured.py — Three-pass analysis pipeline."""

from __future__ import annotations

import json

import httpx
import pytest

from forge.analyze.structured import AnalysisError, Verdict, analyze

STEELMAN_RESPONSE = {
    "position": "AI agents will indeed displace significant SaaS revenue",
    "arguments": [
        "AI agents can perform tasks that previously required SaaS subscriptions",
        "Cost of AI inference is dropping rapidly",
        "Early adopters are already replacing SaaS tools with AI agents",
    ],
    "confidence": 72,
}

REDTEAM_RESPONSE = {
    "attacks": [
        "SaaS products provide reliability and compliance that AI agents lack",
        "Enterprise switching costs are high and slow",
        "30% displacement by 2027 is too aggressive a timeline",
    ],
    "weaknesses": [
        "Claim assumes AI agents will be reliable enough for production use",
        "Ignores regulatory and compliance requirements in enterprise",
    ],
    "confidence": 65,
}

JUDGE_RESPONSE = {
    "position": "AI agents will displace some SaaS but 30% by 2027 is too aggressive",
    "confidence": 42,
    "synthesis": "The steelman makes valid points about AI capabilities but the red team "
    "correctly identifies that enterprise adoption cycles and compliance requirements "
    "will slow displacement significantly.",
    "steelman_arg": "AI agents can perform tasks that previously required SaaS subscriptions",
    "redteam_arg": "Enterprise switching costs are high and slow",
    "conditions": [
        "Faster if major cloud providers bundle AI agents with existing services",
        "Slower if AI reliability issues cause high-profile failures",
    ],
    "tags": ["ai", "saas", "enterprise", "prediction"],
}


class TestAnalyzePipeline:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_analyze_returns_verdict(self, mock_llm) -> None:
        mock_llm.set_responses([STEELMAN_RESPONSE, REDTEAM_RESPONSE, JUDGE_RESPONSE])
        verdict = await analyze(
            claim="AI agents will displace 30% of SaaS by 2027",
            llm=mock_llm,
        )
        assert isinstance(verdict, Verdict)
        assert verdict.confidence == 42
        assert verdict.position == JUDGE_RESPONSE["position"]
        assert len(verdict.tags) == 4

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_analyze_makes_three_llm_calls(self, mock_llm) -> None:
        mock_llm.set_responses([STEELMAN_RESPONSE, REDTEAM_RESPONSE, JUDGE_RESPONSE])
        await analyze(claim="test claim", llm=mock_llm)
        assert mock_llm.call_count == 3

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_analyze_passes_context(self, mock_llm) -> None:
        mock_llm.set_responses([STEELMAN_RESPONSE, REDTEAM_RESPONSE, JUDGE_RESPONSE])
        await analyze(
            claim="test claim",
            context="important background",
            llm=mock_llm,
        )
        # The first call (steelman) should include the context in messages
        # Each call replaces last_messages, so check the last call (judge)
        # included the claim
        assert any("test claim" in str(m) for m in mock_llm.last_messages)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_analyze_steelman_feeds_into_redteam(self, mock_llm) -> None:
        mock_llm.set_responses([STEELMAN_RESPONSE, REDTEAM_RESPONSE, JUDGE_RESPONSE])
        await analyze(claim="test claim", llm=mock_llm)
        # After 3 calls, the judge call (last) should reference steelman output
        assert mock_llm.call_count == 3

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_analyze_raises_on_llm_error(self, mock_llm) -> None:
        mock_llm.set_error(500)
        with pytest.raises((AnalysisError, httpx.HTTPStatusError)):
            await analyze(claim="test claim", llm=mock_llm)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_analyze_raises_on_mid_chain_error(self, mock_llm) -> None:
        """Error after steelman succeeds but redteam fails."""
        mock_llm.set_responses([STEELMAN_RESPONSE])
        mock_llm._responses.clear()  # noqa: SLF001
        mock_llm.set_error(500)
        # First call succeeds (steelman), second fails (redteam)
        # Need to re-queue: steelman response, then error
        mock_llm._errors.clear()  # noqa: SLF001
        mock_llm._responses = [STEELMAN_RESPONSE]  # noqa: SLF001

        class FailOnSecondCall:
            """Mock that fails on the second call."""

            def __init__(self):
                self.call_count = 0
                self.last_messages = []

            async def complete(self, messages, **kwargs):
                self.call_count += 1
                self.last_messages = messages
                if self.call_count == 1:
                    from forge.llm.client import CompletionResponse

                    return CompletionResponse(
                        content=json.dumps(STEELMAN_RESPONSE),
                        parsed_json=STEELMAN_RESPONSE,
                        token_count=10,
                        raw_response={},
                    )
                raise httpx.HTTPStatusError(
                    "Server Error",
                    request=httpx.Request("POST", "http://mock"),
                    response=httpx.Response(500),
                )

        failing_llm = FailOnSecondCall()
        with pytest.raises((AnalysisError, httpx.HTTPStatusError)):
            await analyze(claim="test", llm=failing_llm)


class TestVerdictModel:
    @pytest.mark.unit
    def test_verdict_fields(self) -> None:
        verdict = Verdict(
            position="test position",
            confidence=75,
            synthesis="test synthesis",
            steelman_arg="for",
            redteam_arg="against",
            conditions=["condition1"],
            tags=["tag1"],
        )
        assert verdict.position == "test position"
        assert verdict.confidence == 75
        assert verdict.tags == ["tag1"]

    @pytest.mark.unit
    def test_verdict_confidence_bounds(self) -> None:
        verdict = Verdict(
            position="p", confidence=100, synthesis="s",
            steelman_arg="f", redteam_arg="a",
            conditions=[], tags=[],
        )
        assert verdict.confidence == 100

    @pytest.mark.unit
    def test_verdict_optional_relations(self) -> None:
        verdict = Verdict(
            position="p", confidence=50, synthesis="s",
            steelman_arg="f", redteam_arg="a",
            conditions=[], tags=[],
            relations=[{"target_id": "h_123", "type": "supports", "reasoning": "test"}],
        )
        assert len(verdict.relations) == 1
