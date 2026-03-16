"""Async HTTP client for llama.cpp OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


class ParseError(Exception):
    """Raised when LLM response cannot be parsed as JSON after retry."""

    def __init__(self, raw_content: str, original_error: Exception) -> None:
        self.raw_content = raw_content
        self.original_error = original_error
        super().__init__(f"Failed to parse JSON from LLM response: {raw_content!r}")


@dataclass
class CompletionResponse:
    """Parsed LLM completion response."""

    content: str
    parsed_json: dict | None
    token_count: int
    raw_response: dict


def _extract_json(text: str) -> dict:
    """Extract JSON from text, handling thinking mode output.

    Strips <think>...</think> blocks and finds the first JSON object.
    """
    # Strip think blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Try parsing the cleaned text directly
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Try to find a JSON object in the text
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        return json.loads(match.group())
    raise json.JSONDecodeError("No JSON object found", text, 0)


@dataclass
class LLMClient:
    """Async client for llama.cpp inference."""

    base_url: str
    timeout: float = 120.0
    max_retries: int = 3
    _backoff_base: float = field(default=1.0, repr=False)

    async def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: dict | None = None,
    ) -> CompletionResponse:
        """Send a chat completion request to llama.cpp.

        Args:
            messages: Chat messages in OpenAI format.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            response_format: If {"type": "json_object"}, parse JSON from response.

        Returns:
            CompletionResponse with content and optionally parsed JSON.

        Raises:
            httpx.HTTPStatusError: After max retries on 503.
            ParseError: If JSON parsing fails after retry (when json mode requested).
        """
        payload: dict = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        wants_json = (
            response_format is not None
            and response_format.get("type") == "json_object"
        )

        raw = await self._request_with_retry(payload)
        content = raw["choices"][0]["message"]["content"]
        token_count = raw.get("usage", {}).get("completion_tokens", 0)

        if not wants_json:
            return CompletionResponse(
                content=content,
                parsed_json=None,
                token_count=token_count,
                raw_response=raw,
            )

        # Try to parse JSON; on failure, retry the LLM call once
        try:
            parsed = _extract_json(content)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Malformed JSON from LLM, retrying once: %s", content[:200])
            raw = await self._request_with_retry(payload)
            content = raw["choices"][0]["message"]["content"]
            token_count = raw.get("usage", {}).get("completion_tokens", 0)
            try:
                parsed = _extract_json(content)
            except (json.JSONDecodeError, ValueError) as e:
                raise ParseError(content, e) from e

        return CompletionResponse(
            content=content,
            parsed_json=parsed,
            token_count=token_count,
            raw_response=raw,
        )

    async def _request_with_retry(self, payload: dict) -> dict:
        """Send HTTP request with exponential backoff retry on 503."""
        last_exc: httpx.HTTPStatusError | None = None
        for attempt in range(self.max_retries + 1):
            async with httpx.AsyncClient(timeout=self.timeout) as http:
                resp = await http.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                )
            if resp.status_code == 503 and attempt < self.max_retries:
                wait = self._backoff_base * (2**attempt)
                logger.warning("503 from LLM, retrying in %.1fs (attempt %d)", wait, attempt + 1)
                await asyncio.sleep(wait)
                last_exc = httpx.HTTPStatusError(
                    "Service Unavailable", request=resp.request, response=resp
                )
                continue
            resp.raise_for_status()
            return resp.json()
        raise last_exc  # type: ignore[misc]


@dataclass
class MockLLMClient:
    """Test double for LLMClient. Matches the real client's interface."""

    _responses: list[dict] = field(default_factory=list)
    _errors: list[int] = field(default_factory=list)
    call_count: int = 0
    last_messages: list[dict] = field(default_factory=list)

    def set_response(self, json_dict: dict) -> None:
        """Set a single response for the next call."""
        self._responses = [json_dict]

    def set_responses(self, json_dicts: list[dict]) -> None:
        """Queue multiple responses in order."""
        self._responses = list(json_dicts)

    def set_error(self, status_code: int) -> None:
        """Make the next call raise an HTTP error."""
        self._errors = [status_code]

    async def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: dict | None = None,
    ) -> CompletionResponse:
        """Return queued response or raise queued error."""
        self.call_count += 1
        self.last_messages = messages

        if self._errors:
            status_code = self._errors.pop(0)
            raise httpx.HTTPStatusError(
                f"Mock error {status_code}",
                request=httpx.Request("POST", "http://mock/v1/chat/completions"),
                response=httpx.Response(status_code),
            )

        if not self._responses:
            raise RuntimeError("MockLLMClient: no responses queued")

        data = self._responses.pop(0)
        content = json.dumps(data)
        return CompletionResponse(
            content=content,
            parsed_json=data,
            token_count=len(content.split()),
            raw_response={"mock": True},
        )
