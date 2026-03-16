#!/usr/bin/env python3
"""Validate llama-server inference backend for forge.

Checks health, JSON mode, concurrency, and throughput against a running
llama-server instance on localhost:8080.

Qwen3 uses thinking mode by default — reasoning goes to reasoning_content,
final answer goes to content. The response_format=json_object grammar constraint
blocks thinking tokens, so for structured output with thinking we instruct
the model to return JSON in content and parse it ourselves.

Usage: python scripts/validate_model.py [--base-url http://localhost:8080]
"""

import argparse
import asyncio
import json
import sys
import time

import httpx

DEFAULT_BASE_URL = "http://localhost:8080"
TIMEOUT = 120.0  # seconds per request (model can be slow on first load)
VERBOSE = False


def get_content(choice: dict) -> tuple[str, str]:
    """Extract content and reasoning from a chat completion choice."""
    msg = choice["message"]
    content = msg.get("content", "") or ""
    reasoning = msg.get("reasoning_content", "") or ""
    return content, reasoning


def strip_fences(text: str) -> str:
    """Strip markdown code fences from text."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text


def show_response(content: str, reasoning: str) -> None:
    """Print full response content when verbose mode is enabled."""
    if not VERBOSE:
        return
    if reasoning:
        print(f"    [thinking] {reasoning}")
    if content:
        print(f"    [content]  {content}")
    print()


def result(name: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    return passed


async def check_health(client: httpx.AsyncClient) -> bool:
    r = await client.get("/health")
    if VERBOSE:
        print(f"    [response] {r.text.strip()}\n")
    return result("Health check", r.status_code == 200, f"status={r.status_code}")


async def check_model_loaded(client: httpx.AsyncClient) -> bool:
    r = await client.get("/v1/models")
    if r.status_code != 200:
        return result("Model loaded", False, f"status={r.status_code}")
    data = r.json()
    models = [m["id"] for m in data.get("data", [])]
    if VERBOSE:
        print(f"    [response] {json.dumps(data, indent=2)}\n")
    return result("Model loaded", len(models) > 0, f"models={models}")


async def check_basic_completion(client: httpx.AsyncClient) -> bool:
    r = await client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "Say hello in one sentence."}
            ],
            "max_tokens": 512,
            "temperature": 0.7,
        },
    )
    if r.status_code != 200:
        return result("Basic completion", False, f"status={r.status_code}")
    content, reasoning = get_content(r.json()["choices"][0])
    show_response(content, reasoning)
    has_output = len(content) > 0 or len(reasoning) > 0
    detail = f"content_len={len(content)}, reasoning_len={len(reasoning)}"
    return result("Basic completion", has_output, detail)


async def check_thinking_works(client: httpx.AsyncClient) -> bool:
    """Verify that the model actually uses thinking mode (reasoning_content)."""
    r = await client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "What is 137 * 29? Show your work."}
            ],
            "max_tokens": 1024,
            "temperature": 0.3,
        },
    )
    if r.status_code != 200:
        return result("Thinking mode", False, f"status={r.status_code}")
    content, reasoning = get_content(r.json()["choices"][0])
    show_response(content, reasoning)
    has_thinking = len(reasoning) > 0
    detail = f"reasoning_len={len(reasoning)}, content_len={len(content)}"
    return result("Thinking mode", has_thinking, detail)


async def check_json_with_thinking(client: httpx.AsyncClient) -> bool:
    """Test JSON output with thinking enabled (no response_format constraint).

    This is the production path for forge: the model thinks in reasoning_content,
    then returns clean JSON in content.
    """
    r = await client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": 'Return a JSON object with keys "name" and "age". '
                    "Return ONLY valid JSON, no markdown, no explanation.",
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.3,
        },
    )
    if r.status_code != 200:
        return result("JSON + thinking", False, f"status={r.status_code}")
    content, reasoning = get_content(r.json()["choices"][0])
    show_response(content, reasoning)
    try:
        parsed = json.loads(content.strip())
        has_keys = "name" in parsed and "age" in parsed
        detail = f"parsed={parsed}"
        if reasoning:
            detail += f", thinking_len={len(reasoning)}"
        return result("JSON + thinking", has_keys, detail)
    except json.JSONDecodeError as e:
        return result(
            "JSON + thinking", False, f"parse error: {e}, raw={content[:200]}"
        )


async def check_structured_output(client: httpx.AsyncClient) -> bool:
    """Test forge-style structured output: claim analysis with thinking."""
    r = await client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Analyze the claim: 'The Earth orbits the Sun.' "
                    "Return ONLY a JSON object with these keys: "
                    '"claim" (string), "probability" (float 0-1), '
                    '"reasoning" (string), "confidence" (string: high/medium/low). '
                    "No markdown, no explanation outside the JSON.",
                }
            ],
            "max_tokens": 2048,
            "temperature": 0.3,
        },
    )
    if r.status_code != 200:
        return result("Structured output", False, f"status={r.status_code}")
    content, reasoning = get_content(r.json()["choices"][0])
    show_response(content, reasoning)
    text = strip_fences(content)
    try:
        parsed = json.loads(text)
        required = {"claim", "probability", "reasoning", "confidence"}
        present = required & set(parsed.keys())
        ok = present == required
        detail = f"keys={list(parsed.keys())}"
        if reasoning:
            detail += f", thinking_len={len(reasoning)}"
        return result("Structured output", ok, detail)
    except json.JSONDecodeError as e:
        return result(
            "Structured output", False, f"parse error: {e}, raw={text[:200]}"
        )


async def check_concurrent(client: httpx.AsyncClient) -> bool:
    async def single_request(i: int) -> tuple[int, bool, str, str]:
        r = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": f"What is {i} + {i}? Answer briefly.",
                    }
                ],
                "max_tokens": 256,
                "temperature": 0.3,
            },
        )
        if r.status_code == 200:
            content, reasoning = get_content(r.json()["choices"][0])
            return i, True, content, reasoning
        return i, False, "", ""

    t0 = time.monotonic()
    results = await asyncio.gather(*[single_request(i) for i in range(8)])
    elapsed = time.monotonic() - t0
    successes = sum(1 for _, ok, _, _ in results if ok)
    if VERBOSE:
        for i, ok, content, _reasoning in sorted(results):
            status = "ok" if ok else "FAIL"
            answer = content.strip().replace("\n", " ")[:80] if content else "(empty)"
            print(f"    [slot {i}] ({status}) {answer}")
        print()
    return result(
        "Concurrent requests (8)",
        successes == 8,
        f"{successes}/8 succeeded in {elapsed:.1f}s",
    )


async def check_throughput(client: httpx.AsyncClient) -> bool:
    prompt = (
        "Write a detailed analysis of climate change impacts on agriculture. "
        "Cover at least 5 key points with supporting evidence."
    )
    t0 = time.monotonic()
    r = await client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.7,
        },
    )
    elapsed = time.monotonic() - t0
    if r.status_code != 200:
        return result("Throughput", False, f"status={r.status_code}")
    usage = r.json().get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    tps = completion_tokens / elapsed if elapsed > 0 else 0
    content, reasoning = get_content(r.json()["choices"][0])
    show_response(content, reasoning)
    detail = f"{completion_tokens} tokens in {elapsed:.1f}s ({tps:.1f} tok/s)"
    if reasoning:
        detail += " [thinking enabled]"
    return result("Throughput", completion_tokens > 0, detail)


async def check_json_reliability(client: httpx.AsyncClient) -> bool:
    """Run 10 JSON requests with thinking enabled, count parse failures."""
    failures = 0
    for i in range(10):
        r = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Return ONLY this JSON, nothing else: "
                        f"{{\"index\": {i}, \"value\": \"test_{i}\"}}",
                    }
                ],
                "max_tokens": 1024,
                "temperature": 0.3,
            },
        )
        if r.status_code != 200:
            failures += 1
            if VERBOSE:
                print(f"    [{i}] HTTP FAIL: status={r.status_code}")
            continue
        content, reasoning = get_content(r.json()["choices"][0])
        text = strip_fences(content)
        try:
            parsed = json.loads(text)
            if VERBOSE:
                print(f"    [{i}] {parsed}")
        except json.JSONDecodeError:
            failures += 1
            if VERBOSE:
                r_len = len(reasoning)
                print(f"    [{i}] PARSE FAIL (thinking={r_len}): {text[:100]}")
    if VERBOSE:
        print()
    return result(
        "JSON reliability (10/10)",
        failures == 0,
        f"{10 - failures}/10 parsed OK",
    )


async def main(base_url: str) -> int:
    print("\n=== Forge Model Validation (thinking enabled) ===")
    print(f"Target: {base_url}\n")

    async with httpx.AsyncClient(base_url=base_url, timeout=TIMEOUT) as client:
        # Check server is up first
        try:
            await client.get("/health")
        except httpx.ConnectError:
            print(f"  [FAIL] Cannot connect to {base_url}")
            print("         Is llama-server running?")
            return 1

        checks = [
            check_health,
            check_model_loaded,
            check_basic_completion,
            check_thinking_works,
            check_json_with_thinking,
            check_structured_output,
            check_concurrent,
            check_throughput,
            check_json_reliability,
        ]

        results = []
        for check in checks:
            passed = await check(client)
            results.append(passed)

    passed = sum(results)
    total = len(results)
    print(f"\n=== Results: {passed}/{total} passed ===\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate forge inference backend")
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help="llama-server base URL"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show full response content and reasoning for each check",
    )
    args = parser.parse_args()
    VERBOSE = args.verbose
    sys.exit(asyncio.run(main(args.base_url)))
