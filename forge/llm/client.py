"""Async HTTP client for llama.cpp OpenAI-compatible API."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx


@dataclass
class CompletionResponse:
    """Parsed response from an LLM completion call."""

    content: dict
    token_count: int


class ParseError(Exception):
    """Raised when the LLM returns malformed JSON."""

    def __init__(self, message: str, *, raw_response: str) -> None:
        super().__init__(message)
        self.raw_response = raw_response


@dataclass
class LLMClient:
    """Async client for llama.cpp OpenAI-compatible /v1/chat/completions."""

    base_url: str
    timeout: float = 60.0
    max_retries: int = 3
    call_count: int = field(default=0, init=False)
    last_messages: list[dict] | None = field(default=None, init=False)

    async def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: dict | None = None,
    ) -> CompletionResponse:
        """Send a chat completion request to llama.cpp.

        Retries on 503 with exponential backoff. On malformed JSON,
        retries once then raises ParseError.
        """
        self.last_messages = messages
        self.call_count += 1

        payload: dict = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        raw_text = await self._request_with_retry(payload)
        return self._parse_response(raw_text, messages, temperature, max_tokens, response_format)

    async def _request_with_retry(self, payload: dict) -> str:
        """POST to /v1/chat/completions with retry on 503."""
        import asyncio

        url = f"{self.base_url}/v1/chat/completions"
        last_exc: Exception | None = None

        for attempt in range(self.max_retries):
            async with httpx.AsyncClient(timeout=self.timeout) as http:
                resp = await http.post(url, json=payload)

            if resp.status_code == 503:
                last_exc = httpx.HTTPStatusError(
                    f"503 on attempt {attempt + 1}",
                    request=resp.request,
                    response=resp,
                )
                await asyncio.sleep(2 ** attempt)
                continue

            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        raise last_exc  # type: ignore[misc]

    def _parse_response(
        self,
        raw_text: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        response_format: dict | None,
    ) -> CompletionResponse:
        """Parse JSON from LLM response, retry once on failure."""
        try:
            content = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ParseError(
                f"Malformed JSON from LLM: {raw_text[:200]}",
                raw_response=raw_text,
            ) from exc
        token_count = len(raw_text.split())
        return CompletionResponse(content=content, token_count=token_count)
