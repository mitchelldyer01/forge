# FORGE — Task Tracking

## Phase 0: The Kernel

### P0.1: Database layer (schema + store + models)

**Status:** COMPLETE

**Acceptance Criteria:**
- [x] All tables from FORGE-ARCHITECTURE.md Section 4 created via schema.py
- [x] All indexes from Section 4 created
- [x] WAL mode enabled on file-backed connections
- [x] In-memory DB works for tests
- [x] ULID generation with type prefixes (h_, e_, r_, s_, st_, ap_, p_, a_, f_)
- [x] All timestamps auto-populated as ISO 8601 UTC
- [x] Pydantic models for: Hypothesis, Evidence, Relation, Simulation, AgentPersona, SimulationTurn, Prediction, Feed, Article, Feedback, CalibrationSnapshot
- [x] Store provides typed CRUD methods:
  - Hypothesis: create, get_by_id, update, list_by_status, list_by_confidence_range
  - Evidence: create, get_by_id
  - Relation: create, list_by_source, list_by_target
  - Feedback: create
- [x] All Store methods return Pydantic models, not raw dicts/tuples
- [x] Edge cases handled: missing required fields raise errors, invalid status values rejected

### P0.2: LLM client

**Status:** COMPLETE

**Acceptance Criteria:**
- [x] AsyncClient wraps httpx targeting llama.cpp OpenAI-compatible API
- [x] JSON mode via response_format parameter
- [x] Exponential backoff retry on 503 (max 3 attempts)
- [x] Configurable timeout per call
- [x] Malformed JSON: retry once, then raise ParseError with raw response logged
- [x] MockLLMClient in conftest.py matches the real client's interface exactly
- [x] Token count extraction from response
- [x] Call history tracking (call_count, last_messages)

### P0.3: System prompts

**Status:** COMPLETE

**Acceptance Criteria:**
- [x] Three markdown files: steelman.md, redteam.md, judge.md
- [x] Each defines a clear role, rules, and JSON output schema
- [x] Prompt loader reads files and renders with Jinja2 (claim, context vars)
- [x] Judge prompt includes confidence scoring 0-100

### P0.4: Structured analysis pipeline

**Status:** COMPLETE

**Acceptance Criteria:**
- [x] Three sequential async LLM calls: steelman -> redteam -> judge
- [x] Each call receives output from prior calls as context
- [x] Returns a Verdict model with: position, confidence, steelman_arg, redteam_arg, synthesis, conditions, tags
- [x] Handles LLM failure at any step (logs error, returns partial result or raises)

### P0.5: CLI entrypoint

**Status:** COMPLETE

**Acceptance Criteria:**
- [x] `forge test "claim"` runs structured analysis and prints Rich-formatted verdict
- [x] `forge test "claim" --context "background"` passes context
- [x] `forge test "claim" --json` outputs raw JSON (for piping)
- [x] Non-zero exit code on pipeline failure

### P0.6: Persistence

**Status:** COMPLETE

**Acceptance Criteria:**
- [x] Every `forge test` run persists the hypothesis + verdict to SQLite
- [x] Hypothesis fields populated: claim, context, confidence, source="manual", tags

### P0.7: History command

**Status:** COMPLETE

**Acceptance Criteria:**
- [x] `forge history` shows a Rich table of past hypotheses
- [x] Columns: ID (short), claim (truncated), confidence, status, created_at
- [x] `--status` and `--limit` flags work
- [x] Empty DB prints "No hypotheses yet" (not a crash)

### P0.8: Status command

**Status:** COMPLETE

**Acceptance Criteria:**
- [x] Shows hypothesis count by status, total evidence, total feedback
- [x] Shows LLM URL configuration
- [x] Shows DB path
- [x] Graceful message if DB doesn't exist yet

### P0.9: Test coverage + cleanup

**Status:** NOT STARTED

### P0.10: Manual quality validation

**Status:** NOT STARTED (requires live llama-server)
