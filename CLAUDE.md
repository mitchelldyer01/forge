# FORGE — Calibrated Prediction Engine

## Commands
- `uv run pytest` — run all tests (must pass before any commit)
- `uv run pytest -m unit` — run unit tests only (fast)
- `uv run pytest -m integration` — run integration tests only
- `uv run pytest --cov=forge --cov-report=term-missing` — coverage report
- `uv run ruff check forge/ tests/` — lint (must pass before any commit)
- `uv run forge test "claim"` — test a hypothesis (requires llama-server on :8080)
- `uv run forge status` — system health

## Architecture
Read FORGE-ARCHITECTURE.md for the complete technical specification.
That document is the source of truth for all design decisions.
Do not deviate from it without explicitly stating why.

## TDD Workflow — Red-Green-Refactor (MANDATORY)

Every unit of work follows strict Test-Driven Development. This is not optional.
The cycle is: RED → GREEN → REFACTOR → COMMIT. No exceptions.

### The Cycle

**RED: Write a failing test first.**
- Before writing ANY implementation code, write a test that captures the
  expected behavior.
- Run `uv run pytest` and confirm the test FAILS. If it passes, your test
  is not testing anything new — rewrite it.
- The test failure message should clearly describe what's missing.
- Commit nothing at this stage.

**GREEN: Write the minimum code to make the test pass.**
- Write ONLY enough implementation to make the failing test pass. No more.
- Do not implement features the tests don't ask for yet.
- Do not optimize, refactor, or clean up. Just make it green.
- Run `uv run pytest` and confirm ALL tests pass (not just the new one).

**REFACTOR: Clean up while green.**
- Now improve the code: remove duplication, improve naming, extract helpers.
- Run `uv run pytest` after every change to confirm you haven't broken anything.
- If a test breaks during refactor, undo the last change immediately.
- Refactoring must not change behavior — only structure.

**COMMIT: Lock in the work.**
- Run `uv run pytest` (all pass) AND `uv run ruff check forge/ tests/` (clean).
- Commit with a descriptive message.
- Only commit when green. Never commit red.

### TDD Rules

1. **No production code without a failing test.** If you catch yourself writing
   forge/ code without a corresponding test in tests/, stop and write the test first.

2. **One behavior per test.** Each test function tests exactly one behavior.
   Name it `test_{module}_{behavior}_{expected_outcome}`.
   Example: `test_store_save_hypothesis_assigns_ulid`

3. **Tests are documentation.** A new developer should understand what the code
   does by reading the test names alone. Use descriptive names, not `test_1`, `test_2`.

4. **Arrange-Act-Assert.** Every test follows this structure:
```python
   def test_something():
       # Arrange: set up preconditions
       store = Store(":memory:")
       
       # Act: perform the operation
       result = store.save_hypothesis(claim="test claim")
       
       # Assert: verify the outcome
       assert result.id.startswith("h_")
       assert result.claim == "test claim"
```

5. **Test the interface, not the implementation.** Tests call public methods
   and assert on return values or state changes. Never test private methods
   or internal data structures directly.

6. **Fixtures over setup duplication.** Shared test state goes in conftest.py
   as pytest fixtures. No copy-pasting setup code between test files.

7. **Mock at the boundary.** Mock the LLM client and external I/O. Never mock
   internal modules — if you need to mock something internal, the design is
   wrong and needs refactoring.

8. **Each test file mirrors a source file.** `forge/db/store.py` → `tests/test_store.py`.
   `forge/llm/client.py` → `tests/test_llm_client.py`.
   `forge/analyze/structured.py` → `tests/test_structured.py`.

9. **Edge cases are first-class.** For every happy path test, write at least one:
   - Empty/null input test
   - Malformed input test (especially for LLM JSON parsing)
   - Boundary condition test

10. **No tests that test the framework.** Don't test that SQLite works or that
    Pydantic validates. Test YOUR code's behavior.

### Task-Level TDD Flow

For each backlog task (e.g., P0.1), the workflow is:

```
SPEC: Write acceptance criteria in TASKS.md
RED: Write all tests for this task's acceptance criteria
→ Run pytest → All new tests FAIL (existing tests still pass)
GREEN: Implement until all tests pass
→ Run pytest → ALL tests pass
REFACTOR: Clean up implementation while staying green
→ Run pytest after each change → ALL tests pass
LINT: Run ruff check → clean
COMMIT: phase0: P0.1 — SQLite schema + store + models
```

## Code Standards
- Python 3.12+. Type hints on all function signatures.
- No ORM. Raw SQL in forge/db/store.py. Pydantic models in forge/db/models.py.
- All IDs are ULIDs with type prefixes: h_, e_, r_, s_, st_, ap_, p_, a_, f_.
- All timestamps are ISO 8601 UTC strings.
- All LLM outputs are JSON. Use response_format={"type": "json_object"}.
- Handle malformed LLM JSON gracefully (retry once, then log and skip).
- All LLM calls go through forge/llm/client.py. Never call httpx directly elsewhere.
- Prompts live in markdown files under relevant prompts/ dirs. Never hardcode prompt text.
- Tests use MockLLMClient (defined in tests/conftest.py). Never hit real inference.
- One function, one job. Files under 200 lines. Split when larger.
- Use `async def` for anything involving LLM calls or I/O.

## Test Infrastructure (tests/conftest.py)

The conftest.py file must provide these fixtures:

- `db` — fresh in-memory SQLite Store instance, schema applied, torn down after test
- `mock_llm` — MockLLMClient that returns configurable JSON responses
- `sample_hypothesis` — a pre-built Hypothesis model for reuse
- `sample_evidence` — a pre-built Evidence model for reuse

MockLLMClient must support:
- `set_response(json_dict)` — next call returns this JSON
- `set_responses([json_dict, ...])` — queue multiple responses in order
- `set_error(status_code)` — next call raises an HTTP error
- `call_count` — how many times complete() was called
- `last_messages` — the messages from the most recent call

## Dependencies
Only add dependencies listed in FORGE-ARCHITECTURE.md Section 13.
If you need something not listed, state why before adding.

## Git
- Commit after each completed task (not mid-task).
- Message format: `phase0: P0.X — short description`
- Run pytest + ruff before every commit. Do not commit with failures.
- Never commit with a red test. If you can't make it green, revert.
