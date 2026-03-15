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

**Status:** IN PROGRESS

**Acceptance Criteria:**
- [ ] AsyncClient wraps httpx targeting llama.cpp OpenAI-compatible API
- [ ] JSON mode via response_format parameter
- [ ] Exponential backoff retry on 503 (max 3 attempts)
- [ ] Configurable timeout per call
- [ ] Malformed JSON: retry once, then raise ParseError with raw response logged
- [ ] MockLLMClient in conftest.py matches the real client's interface exactly
- [ ] Token count extraction from response
- [ ] Call history tracking (call_count, last_messages)

### P0.3: System prompts

**Status:** NOT STARTED

**Acceptance Criteria:**
- [ ] Three markdown files: steelman.md, redteam.md, judge.md
- [ ] Each defines a clear role, rules, and JSON output schema
- [ ] Prompt loader reads files and renders with Jinja2 (claim, context vars)
- [ ] Judge prompt includes confidence scoring 0-100
