"""Tests for forge/llm/client.py — LLM client with retry and JSON parsing."""

from __future__ import annotations

import json

import httpx
import pytest

from forge.llm.client import CompletionResponse, LLMClient, ParseError, _extract_json


@pytest.fixture
def llm_url() -> str:
    return "http://127.0.0.1:8080"


@pytest.fixture
def client(llm_url: str) -> LLMClient:
    return LLMClient(base_url=llm_url, timeout=5.0)


class TestCompletionResponse:
    @pytest.mark.unit
    def test_completion_response_fields(self) -> None:
        resp = CompletionResponse(
            content="hello",
            parsed_json=None,
            token_count=10,
            raw_response={"choices": []},
        )
        assert resp.content == "hello"
        assert resp.parsed_json is None
        assert resp.token_count == 10

    @pytest.mark.unit
    def test_completion_response_with_parsed_json(self) -> None:
        data = {"position": "support", "confidence": 75}
        resp = CompletionResponse(
            content=json.dumps(data),
            parsed_json=data,
            token_count=20,
            raw_response={},
        )
        assert resp.parsed_json == data


class TestLLMClientInit:
    @pytest.mark.unit
    def test_client_stores_base_url(self, client: LLMClient, llm_url: str) -> None:
        assert client.base_url == llm_url

    @pytest.mark.unit
    def test_client_stores_timeout(self, client: LLMClient) -> None:
        assert client.timeout == 5.0


class TestLLMClientComplete:
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_complete_returns_completion_response(
        self, client: LLMClient, httpx_mock
    ) -> None:
        httpx_mock.add_response(
            url=f"{client.base_url}/v1/chat/completions",
            json={
                "choices": [{"message": {"content": '{"result": "ok"}'}}],
                "usage": {"completion_tokens": 15},
            },
        )
        resp = await client.complete(
            messages=[{"role": "user", "content": "test"}],
            response_format={"type": "json_object"},
        )
        assert isinstance(resp, CompletionResponse)
        assert resp.parsed_json == {"result": "ok"}
        assert resp.token_count == 15

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_complete_parses_json_from_content(
        self, client: LLMClient, httpx_mock
    ) -> None:
        json_data = {"position": "support", "confidence": 80}
        httpx_mock.add_response(
            url=f"{client.base_url}/v1/chat/completions",
            json={
                "choices": [{"message": {"content": json.dumps(json_data)}}],
                "usage": {"completion_tokens": 10},
            },
        )
        resp = await client.complete(
            messages=[{"role": "user", "content": "analyze"}],
            response_format={"type": "json_object"},
        )
        assert resp.parsed_json == json_data

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_complete_extracts_json_from_thinking_content(
        self, client: LLMClient, httpx_mock
    ) -> None:
        """When model uses thinking mode, JSON may be embedded after <think> tags."""
        content = '<think>Let me reason...</think>\n{"result": "analyzed"}'
        httpx_mock.add_response(
            url=f"{client.base_url}/v1/chat/completions",
            json={
                "choices": [{"message": {"content": content}}],
                "usage": {"completion_tokens": 25},
            },
        )
        resp = await client.complete(
            messages=[{"role": "user", "content": "test"}],
            response_format={"type": "json_object"},
        )
        assert resp.parsed_json == {"result": "analyzed"}

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_complete_retries_on_503(
        self, client: LLMClient, httpx_mock
    ) -> None:
        httpx_mock.add_response(
            url=f"{client.base_url}/v1/chat/completions",
            status_code=503,
        )
        httpx_mock.add_response(
            url=f"{client.base_url}/v1/chat/completions",
            json={
                "choices": [{"message": {"content": '{"ok": true}'}}],
                "usage": {"completion_tokens": 5},
            },
        )
        resp = await client.complete(
            messages=[{"role": "user", "content": "test"}],
            response_format={"type": "json_object"},
        )
        assert resp.parsed_json == {"ok": True}

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_complete_raises_after_max_retries(
        self, client: LLMClient, httpx_mock
    ) -> None:
        for _ in range(4):
            httpx_mock.add_response(
                url=f"{client.base_url}/v1/chat/completions",
                status_code=503,
            )
        with pytest.raises(httpx.HTTPStatusError):
            await client.complete(
                messages=[{"role": "user", "content": "test"}],
            )

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_complete_malformed_json_retries_once(
        self, client: LLMClient, httpx_mock
    ) -> None:
        httpx_mock.add_response(
            url=f"{client.base_url}/v1/chat/completions",
            json={
                "choices": [{"message": {"content": "not json at all {"}}],
                "usage": {"completion_tokens": 5},
            },
        )
        httpx_mock.add_response(
            url=f"{client.base_url}/v1/chat/completions",
            json={
                "choices": [{"message": {"content": '{"fixed": true}'}}],
                "usage": {"completion_tokens": 5},
            },
        )
        resp = await client.complete(
            messages=[{"role": "user", "content": "test"}],
            response_format={"type": "json_object"},
        )
        assert resp.parsed_json == {"fixed": True}

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_complete_malformed_json_raises_parse_error(
        self, client: LLMClient, httpx_mock
    ) -> None:
        for _ in range(2):
            httpx_mock.add_response(
                url=f"{client.base_url}/v1/chat/completions",
                json={
                    "choices": [{"message": {"content": "bad json"}}],
                    "usage": {"completion_tokens": 5},
                },
            )
        with pytest.raises(ParseError) as exc_info:
            await client.complete(
                messages=[{"role": "user", "content": "test"}],
                response_format={"type": "json_object"},
            )
        assert "bad json" in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_complete_http_error_includes_response_body(
        self, client: LLMClient, httpx_mock
    ) -> None:
        """HTTP errors include the server's response body for debugging."""
        httpx_mock.add_response(
            url=f"{client.base_url}/v1/chat/completions",
            status_code=500,
            text='{"error": "model not loaded, try again later"}',
        )
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await client.complete(
                messages=[{"role": "user", "content": "test"}],
            )
        assert "model not loaded" in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_complete_without_json_mode_skips_parsing(
        self, client: LLMClient, httpx_mock
    ) -> None:
        httpx_mock.add_response(
            url=f"{client.base_url}/v1/chat/completions",
            json={
                "choices": [{"message": {"content": "plain text response"}}],
                "usage": {"completion_tokens": 5},
            },
        )
        resp = await client.complete(
            messages=[{"role": "user", "content": "test"}],
        )
        assert resp.content == "plain text response"
        assert resp.parsed_json is None

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_complete_sends_temperature_and_max_tokens(
        self, client: LLMClient, httpx_mock
    ) -> None:
        httpx_mock.add_response(
            url=f"{client.base_url}/v1/chat/completions",
            json={
                "choices": [{"message": {"content": '{"ok": true}'}}],
                "usage": {"completion_tokens": 5},
            },
        )
        await client.complete(
            messages=[{"role": "user", "content": "test"}],
            temperature=0.3,
            max_tokens=1024,
        )
        request = httpx_mock.get_request()
        body = json.loads(request.content)
        assert body["temperature"] == 0.3
        assert body["max_tokens"] == 1024


class TestExtractJsonTruncated:
    """Tests for _extract_json handling truncated JSON from LLM."""

    @pytest.mark.unit
    def test_extract_json_truncated_array_recovers_complete_objects(self) -> None:
        """Truncated JSON with complete objects before the cut should recover them."""
        truncated = (
            '{"agents": [{"archetype": "analyst", "name": "Alice"}, '
            '{"archetype": "founder", "name": "Bob"}, {"archetype": "regul'
        )
        result = _extract_json(truncated)
        assert len(result["agents"]) == 2
        assert result["agents"][0]["name"] == "Alice"
        assert result["agents"][1]["name"] == "Bob"

    @pytest.mark.unit
    def test_extract_json_truncated_mid_object_recovers_prior(self) -> None:
        """Truncation mid-object recovers all complete prior objects."""
        truncated = '{"agents": [{"archetype": "a", "name": "X"}, {"archetype": "b", "name":'
        result = _extract_json(truncated)
        assert len(result["agents"]) == 1
        assert result["agents"][0]["name"] == "X"

    @pytest.mark.unit
    def test_extract_json_truncated_no_complete_objects_raises(self) -> None:
        """Truncation before any complete object raises JSONDecodeError."""
        truncated = '{"agents": [{"archetype": "regul'
        with pytest.raises(json.JSONDecodeError):
            _extract_json(truncated)

    @pytest.mark.unit
    def test_extract_json_valid_json_not_affected(self) -> None:
        """Valid JSON passes through without modification."""
        valid = '{"agents": [{"archetype": "a"}, {"archetype": "b"}]}'
        result = _extract_json(valid)
        assert len(result["agents"]) == 2


class TestParseErrorMessage:
    """ParseError should include actionable diagnostic information."""

    @pytest.mark.unit
    def test_parse_error_includes_truncated_content(self) -> None:
        """ParseError message shows first 200 chars of raw LLM output."""
        raw = "This is not JSON " * 20  # long content
        err = ParseError(raw, json.JSONDecodeError("Expecting value", raw, 0))
        msg = str(err)
        assert "LLM returned non-JSON" in msg
        assert raw[:200] in msg

    @pytest.mark.unit
    def test_parse_error_includes_original_error_type(self) -> None:
        """ParseError message includes the type of the original parse failure."""
        raw = "garbage"
        original = json.JSONDecodeError("Expecting value", raw, 0)
        err = ParseError(raw, original)
        msg = str(err)
        assert "JSONDecodeError" in msg

    @pytest.mark.unit
    def test_parse_error_preserves_raw_content_attr(self) -> None:
        """ParseError stores raw_content for programmatic access."""
        raw = "some raw text"
        err = ParseError(raw, ValueError("bad"))
        assert err.raw_content == raw

    @pytest.mark.unit
    def test_parse_error_empty_content_says_so(self) -> None:
        """ParseError with empty content explicitly says 'empty response'."""
        err = ParseError("", json.JSONDecodeError("Expecting value", "", 0))
        msg = str(err)
        assert "empty response" in msg.lower()


class TestMockLLMClient:
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_mock_set_response(self, mock_llm) -> None:
        mock_llm.set_response({"position": "support", "confidence": 75})
        resp = await mock_llm.complete(
            messages=[{"role": "user", "content": "test"}],
        )
        assert resp.parsed_json == {"position": "support", "confidence": 75}

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_mock_set_responses_queue(self, mock_llm) -> None:
        mock_llm.set_responses([
            {"step": "steelman"},
            {"step": "redteam"},
            {"step": "judge"},
        ])
        r1 = await mock_llm.complete(messages=[{"role": "user", "content": "1"}])
        r2 = await mock_llm.complete(messages=[{"role": "user", "content": "2"}])
        r3 = await mock_llm.complete(messages=[{"role": "user", "content": "3"}])
        assert r1.parsed_json == {"step": "steelman"}
        assert r2.parsed_json == {"step": "redteam"}
        assert r3.parsed_json == {"step": "judge"}

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_mock_set_error(self, mock_llm) -> None:
        mock_llm.set_error(503)
        with pytest.raises(httpx.HTTPStatusError):
            await mock_llm.complete(
                messages=[{"role": "user", "content": "test"}],
            )

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_mock_call_count(self, mock_llm) -> None:
        mock_llm.set_responses([{"a": 1}, {"b": 2}])
        assert mock_llm.call_count == 0
        await mock_llm.complete(messages=[{"role": "user", "content": "1"}])
        assert mock_llm.call_count == 1
        await mock_llm.complete(messages=[{"role": "user", "content": "2"}])
        assert mock_llm.call_count == 2

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_mock_last_messages(self, mock_llm) -> None:
        mock_llm.set_response({"ok": True})
        msgs = [{"role": "user", "content": "hello"}]
        await mock_llm.complete(messages=msgs)
        assert mock_llm.last_messages == msgs
