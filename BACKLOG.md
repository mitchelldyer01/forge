# FORGE Backlog

## Phase 0: The Kernel [COMPLETE]
The atomic unit of value. A CLI that takes a claim, stress-tests it through
structured analysis (steelman -> redteam -> judge), and returns a verdict.

### Tasks

**P0.1: Database layer (schema + store + models)**
TDD sequence:
1. RED: Write tests for — schema creation, hypothesis CRUD (create, read, update,
   list by status, list by confidence range), evidence CRUD, relation CRUD,
   feedback CRUD. Test ULID generation, timestamp auto-population, WAL mode.
   Test edge cases: duplicate IDs, missing required fields, invalid status values.
2. GREEN: Implement schema.py, models.py, store.py.
3. REFACTOR: Extract shared query patterns if any.
Acceptance criteria:
- [ ] All tables from FORGE-ARCHITECTURE.md Section 4 exist
- [ ] Store provides typed methods for all CRUD operations
- [ ] All methods return Pydantic models, not raw dicts/tuples
- [ ] In-memory DB works for tests, file-backed DB works for production
- [ ] WAL mode enabled on file-backed connections

**P0.2: LLM client**
TDD sequence:
1. RED: Write tests for — successful completion (mock HTTP), JSON parsing of
   response, retry on 503, timeout handling, malformed JSON handling (retry once
   then raise), token count extraction, call history tracking.
2. GREEN: Implement client.py with async httpx.
3. REFACTOR: Extract retry logic if complex.
Acceptance criteria:
- [ ] AsyncClient wraps httpx targeting llama.cpp OpenAI-compatible API
- [ ] JSON mode via response_format parameter
- [ ] Exponential backoff retry on 503 (max 3 attempts)
- [ ] Configurable timeout per call
- [ ] Malformed JSON: retry once, then raise ParseError with raw response logged
- [ ] MockLLMClient in conftest.py matches the real client's interface exactly

**P0.3: System prompts**
TDD sequence:
1. RED: Write tests for — prompt files exist and are loadable, prompt template
   rendering with claim/context variables, rendered prompts contain required
   structural elements (JSON output instruction, role definition).
2. GREEN: Create steelman.md, redteam.md, judge.md. Write prompt loader.
3. REFACTOR: Extract common prompt structure if shared across files.
Acceptance criteria:
- [ ] Three markdown files: steelman.md, redteam.md, judge.md
- [ ] Each defines a clear role, rules, and JSON output schema
- [ ] Prompt loader reads files and renders with Jinja2 (claim, context vars)
- [ ] Judge prompt includes confidence scoring 0-100

**P0.4: Structured analysis pipeline**
TDD sequence:
1. RED: Write tests for — full pipeline (steelman -> redteam -> judge) using
   MockLLMClient with queued responses. Test that steelman output feeds into
   redteam context, and both feed into judge. Test verdict structure matches
   expected schema. Test pipeline handles LLM error mid-chain gracefully.
2. GREEN: Implement structured.py orchestrating three sequential LLM calls.
3. REFACTOR: Extract shared prompt-assembly logic.
Acceptance criteria:
- [ ] Three sequential async LLM calls: steelman -> redteam -> judge
- [ ] Each call receives output from prior calls as context
- [ ] Returns a Verdict model with: position, confidence, steelman_arg,
      redteam_arg, synthesis, conditions, tags
- [ ] Handles LLM failure at any step (logs error, returns partial result or raises)

**P0.5: CLI entrypoint**
TDD sequence:
1. RED: Write tests for — `forge test "claim"` invokes pipeline with correct args,
   output renders via Rich without crashing, `--context` flag passes context
   through, `--json` flag outputs raw JSON. Use typer.testing.CliRunner.
2. GREEN: Implement cli.py with Typer.
3. REFACTOR: Extract output formatting.
Acceptance criteria:
- [ ] `forge test "claim"` runs structured analysis and prints Rich-formatted verdict
- [ ] `forge test "claim" --context "background"` passes context
- [ ] `forge test "claim" --json` outputs raw JSON (for piping)
- [ ] Non-zero exit code on pipeline failure

**P0.6: Persistence**
TDD sequence:
1. RED: Write tests for — after `forge test`, hypothesis exists in DB with correct
   fields, verdict stored, confidence set from judge output. Test that running
   multiple claims creates multiple hypotheses.
2. GREEN: Wire pipeline output into store.save_hypothesis().
3. REFACTOR: Ensure transaction boundaries are correct.
Acceptance criteria:
- [ ] Every `forge test` run persists the hypothesis + verdict to SQLite
- [ ] Hypothesis fields populated: claim, context, confidence, source="manual", tags

**P0.7: History command**
TDD sequence:
1. RED: Write tests for — `forge history` lists stored hypotheses sorted by
   created_at desc, shows claim/confidence/status, `--status alive` filters,
   `--limit N` limits results. Test empty DB returns clean message.
2. GREEN: Implement history subcommand.
3. REFACTOR: Extract table rendering.
Acceptance criteria:
- [ ] `forge history` shows a Rich table of past hypotheses
- [ ] Columns: ID (short), claim (truncated), confidence, status, created_at
- [ ] `--status` and `--limit` flags work
- [ ] Empty DB prints "No hypotheses yet" (not a crash)

**P0.8: Status command**
TDD sequence:
1. RED: Write tests for — `forge status` shows DB stats (hypothesis count by
   status, total evidence, total debates), llama.cpp health (mock HTTP to
   health endpoint). Test llama.cpp unreachable shows clear error.
2. GREEN: Implement status subcommand.
3. REFACTOR: None expected.
Acceptance criteria:
- [ ] Shows hypothesis count by status, total evidence, total feedback
- [ ] Pings llama.cpp /v1/models and shows model name + status
- [ ] Graceful message if llama.cpp is unreachable

**P0.9: Test coverage + cleanup**
TDD sequence: N/A — this is a review/hardening task.
1. Run `uv run pytest --cov=forge --cov-report=term-missing`
2. Identify untested code paths. Write tests for them (RED -> GREEN).
3. Ensure all edge cases from P0.1-P0.8 are covered.
4. Run full suite. All green. Commit.
Acceptance criteria:
- [ ] Coverage >= 85% on forge/ (excluding __init__.py)
- [ ] No untested error handling paths
- [ ] All tests have descriptive names following the naming convention
- [ ] conftest.py fixtures are documented

**P0.10: Manual quality validation**
Not a code task. Run 10 real claims against live llama.cpp. Score 1-5.
Target: average >= 3.5, at least 3 scores of 4+. See Step 6 below.

### Done When
- All tests green, lint clean, coverage >= 85%
- 10 real claims scored, average >= 3.5
- All tasks checked off in this file
- Git log shows one commit per task with clean messages

## Phase 1: Memory & Retrieval [COMPLETE]
- [x] P1.1: Embedding generation (sentence-transformers, store as BLOB)
- [x] P1.2: Similarity search over hypothesis graph
- [x] P1.3: Inject top-3 related prior hypotheses into analysis context
- [x] P1.4: Judge extracts relations (supports/contradicts/refines) to existing hypotheses
- [x] P1.5: `forge graph <topic>` — show related hypotheses as a tree
- [x] P1.6: Tests for retrieval + relation extraction
- [x] P1.7: Coverage >= 85% on new code (achieved 93%, 117 tests)

## Phase 2: Swarm Simulation [COMPLETE]
- [x] P2.0: Store CRUD for simulations, personas, turns, predictions
- [x] P2.1: Agent persona generation from seed material
- [x] P2.2: Arena — multi-round simulation loop with async orchestration
- [x] P2.3: Interaction selection (opposing views, domain experts, contrarians)
- [x] P2.4: Consensus extraction (computational, no LLM)
- [x] P2.5: Prediction extraction from simulation output
- [x] P2.6: `forge simulate "scenario"` CLI command with `--agents N` flag
- [x] P2.7: Upgrade path: flag high-confidence claims for swarm (no scheduler)
- [x] P2.8: Coverage >= 85% on new code (achieved 91% on swarm, 93% overall, 211 tests)

## Phase 3: Continuous Ingestion + Calibration
- [ ] P3.1: RSS feed polling + article storage
- [ ] P3.2: Claim extraction from articles
- [ ] P3.3: Relevance filtering against existing graph
- [ ] P3.4: Pipeline runner (cron-like scheduling)
- [ ] P3.5: Resolution tracking (manual + automated)
- [ ] P3.6: Calibration scoring (Brier score, per-topic, per-archetype)
- [ ] P3.7: Drift detection
- [ ] P3.8: Coverage >= 85% on new code

## Phase 4: Evolution & Output
- [ ] P4.1: Hypothesis selection (cull/promote/fork)
- [ ] P4.2: Agent population evolution
- [ ] P4.3: Obsidian export
- [ ] P4.4: Daily digest generation
- [ ] P4.5: Human feedback loop (endorse/reject/resolve)
- [ ] P4.6: FastAPI server
- [ ] P4.7: Track record page
- [ ] P4.8: Coverage >= 85% on new code

## Phase 5: Hardening
- [ ] P5.1: Dual model routing
- [ ] P5.2: Monitoring + health checks
- [ ] P5.3: Automated resolution via web search
- [ ] P5.4: Backup + export
- [ ] P5.5: Coverage >= 90% overall
