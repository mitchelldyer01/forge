"""
Tests for forge.llm.client — async LLM client for llama.cpp.

Mirrors: forge/llm/client.py
"""

import pytest

from forge.llm.client import CompletionResponse, LLMClient, ParseError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def llm_url() -> str:
    return "http://127.0.0.1:8080"


# ---------------------------------------------------------------------------
# CompletionResponse model
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCompletionResponse:
    def test_completion_response_stores_content(self):
        """CompletionResponse holds the parsed content."""
        resp = CompletionResponse(content={"answer": "yes"}, token_count=10)
        assert resp.content == {"answer": "yes"}

    def test_completion_response_stores_token_count(self):
        """CompletionResponse tracks token usage."""
        resp = CompletionResponse(content={"x": 1}, token_count=42)
        assert resp.token_count == 42


# ---------------------------------------------------------------------------
# LLMClient construction
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLLMClientInit:
    def test_client_stores_base_url(self, llm_url: str):
        """Client stores the base URL for llama.cpp."""
        client = LLMClient(base_url=llm_url)
        assert client.base_url == llm_url

    def test_client_default_timeout(self, llm_url: str):
        """Client has a sensible default timeout."""
        client = LLMClient(base_url=llm_url)
        assert client.timeout > 0

    def test_client_custom_timeout(self, llm_url: str):
        """Client accepts a custom timeout."""
        client = LLMClient(base_url=llm_url, timeout=120.0)
        assert client.timeout == 120.0

    def test_client_tracks_call_count(self, llm_url: str):
        """Client starts with zero call count."""
        client = LLMClient(base_url=llm_url)
        assert client.call_count == 0

    def test_client_last_messages_initially_none(self, llm_url: str):
        """Client starts with no last_messages."""
        client = LLMClient(base_url=llm_url)
        assert client.last_messages is None


# ---------------------------------------------------------------------------
# LLMClient.complete — success cases (using MockLLMClient)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLLMClientComplete:
    @pytest.mark.asyncio
    async def test_complete_returns_completion_response(self):
        """complete() returns a CompletionResponse with parsed JSON."""
        from tests.conftest import MockLLMClient

        mock = MockLLMClient()
        mock.set_response({"verdict": "true", "confidence": 85})

        messages = [{"role": "user", "content": "Test claim"}]
        result = await mock.complete(messages)

        assert isinstance(result, CompletionResponse)
        assert result.content == {"verdict": "true", "confidence": 85}

    @pytest.mark.asyncio
    async def test_complete_increments_call_count(self):
        """Each complete() call increments call_count."""
        from tests.conftest import MockLLMClient

        mock = MockLLMClient()
        mock.set_response({"ok": True})

        await mock.complete([{"role": "user", "content": "one"}])
        assert mock.call_count == 1

        await mock.complete([{"role": "user", "content": "two"}])
        assert mock.call_count == 2

    @pytest.mark.asyncio
    async def test_complete_stores_last_messages(self):
        """complete() stores the messages from the most recent call."""
        from tests.conftest import MockLLMClient

        mock = MockLLMClient()
        mock.set_response({"ok": True})

        messages = [
            {"role": "system", "content": "You are a judge."},
            {"role": "user", "content": "test"},
        ]
        await mock.complete(messages)

        assert mock.last_messages == messages

    @pytest.mark.asyncio
    async def test_complete_with_queued_responses(self):
        """set_responses queues multiple responses returned in order."""
        from tests.conftest import MockLLMClient

        mock = MockLLMClient()
        mock.set_responses([
            {"step": "steelman", "arg": "strong"},
            {"step": "redteam", "arg": "weak"},
            {"step": "judge", "verdict": "true"},
        ])

        r1 = await mock.complete([{"role": "user", "content": "1"}])
        assert r1.content["step"] == "steelman"

        r2 = await mock.complete([{"role": "user", "content": "2"}])
        assert r2.content["step"] == "redteam"

        r3 = await mock.complete([{"role": "user", "content": "3"}])
        assert r3.content["step"] == "judge"

    @pytest.mark.asyncio
    async def test_complete_returns_token_count(self):
        """CompletionResponse includes token count."""
        from tests.conftest import MockLLMClient

        mock = MockLLMClient()
        mock.set_response({"ok": True})

        result = await mock.complete([{"role": "user", "content": "test"}])
        assert isinstance(result.token_count, int)
        assert result.token_count >= 0


# ---------------------------------------------------------------------------
# LLMClient.complete — error cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLLMClientErrors:
    @pytest.mark.asyncio
    async def test_complete_raises_on_http_error(self):
        """set_error causes complete() to raise a RuntimeError."""
        from tests.conftest import MockLLMClient

        mock = MockLLMClient()
        mock.set_error(500)

        with pytest.raises(RuntimeError):
            await mock.complete([{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_complete_error_still_increments_call_count(self):
        """Even failed calls increment call_count for tracking."""
        from tests.conftest import MockLLMClient

        mock = MockLLMClient()
        mock.set_error(500)

        with pytest.raises(RuntimeError):
            await mock.complete([{"role": "user", "content": "test"}])

        assert mock.call_count == 1


# ---------------------------------------------------------------------------
# ParseError
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestParseError:
    def test_parse_error_stores_raw_response(self):
        """ParseError includes the raw response text for debugging."""
        err = ParseError("bad json", raw_response="not valid json {{{")
        assert err.raw_response == "not valid json {{{"
        assert "bad json" in str(err)
