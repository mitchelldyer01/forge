"""
Shared test fixtures for FORGE.

All tests use MockLLMClient. No test ever hits a real LLM endpoint.
DB fixtures provide a fresh in-memory SQLite instance per test.
"""

from __future__ import annotations

import json

import pytest

from forge.db.store import Store
from forge.llm.client import CompletionResponse

# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db() -> Store:
    """Fresh in-memory Store with schema applied, torn down after test."""
    store = Store(":memory:")
    yield store
    store.conn.close()


# ---------------------------------------------------------------------------
# MockLLMClient
# ---------------------------------------------------------------------------

class MockLLMClient:
    """Test double for LLMClient.

    Supports:
    - set_response(json_dict) — next call returns this JSON
    - set_responses([json_dict, ...]) — queue multiple responses in order
    - set_error(status_code) — next call raises an HTTP error
    - call_count — how many times complete() was called
    - last_messages — the messages from the most recent call
    """

    def __init__(self) -> None:
        self.base_url: str = "http://mock:8080"
        self.timeout: float = 60.0
        self.call_count: int = 0
        self.last_messages: list[dict] | None = None
        self._responses: list[dict] = []
        self._default_response: dict | None = None
        self._error: int | None = None

    def set_response(self, json_dict: dict) -> None:
        """Set a default response returned for all subsequent calls."""
        self._default_response = json_dict
        self._responses = []

    def set_responses(self, json_dicts: list[dict]) -> None:
        """Queue multiple responses returned in order."""
        self._responses = list(json_dicts)
        self._default_response = None

    def set_error(self, status_code: int) -> None:
        """Next call raises an HTTP error with this status code."""
        self._error = status_code

    async def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: dict | None = None,
    ) -> CompletionResponse:
        """Mock implementation of LLMClient.complete()."""
        self.last_messages = messages
        self.call_count += 1

        if self._error is not None:
            code = self._error
            self._error = None
            raise RuntimeError(f"MockLLMClient HTTP error: {code}")

        if self._responses:
            response_data = self._responses.pop(0)
        elif self._default_response is not None:
            response_data = self._default_response
        else:
            raise RuntimeError("MockLLMClient: no response configured")
        raw = json.dumps(response_data)
        token_count = len(raw.split())
        return CompletionResponse(content=response_data, token_count=token_count)


@pytest.fixture
def mock_llm() -> MockLLMClient:
    """MockLLMClient instance for testing LLM interactions."""
    return MockLLMClient()
