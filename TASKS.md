# FORGE — Task Tracking

## Phase 0: The Kernel

### P0.1: Database layer (schema + store + models)
**Status:** COMPLETE

### P0.2: Config + LLM client
**Status:** COMPLETE

### P0.3: System prompts
**Status:** COMPLETE

### P0.4: Structured analysis pipeline
**Status:** COMPLETE

### P0.5+P0.6: CLI entrypoint + persistence
**Status:** COMPLETE

### P0.7: History command
**Status:** COMPLETE

### P0.8: Status command
**Status:** COMPLETE

## Phase 1: Memory & Retrieval

### P1.1: Embedding generation
**Status:** COMPLETE

### P1.2: Similarity search
**Status:** COMPLETE

### P1.3: Context injection
**Status:** COMPLETE

### P1.4: Relation extraction in judge
**Status:** COMPLETE

### P1.5: Graph command
**Status:** COMPLETE

### P1.6-P1.7: Test coverage
**Status:** COMPLETE — 93% coverage (target: 85%), 117 tests

## Phase 2: Swarm Simulation

### P2.0: Store CRUD for Swarm Entities
**Status:** NOT STARTED
**Depends on:** Nothing (foundation task)
**Modify:** `forge/db/store.py`, `tests/test_store.py`
**Acceptance criteria:**
- [ ] `save_simulation()`, `get_simulation()`, `update_simulation()`, `list_simulations()` work
- [ ] `save_agent_persona()`, `get_agent_persona()`, `update_agent_persona()`, `list_agent_personas()` work
- [ ] `save_simulation_turn()`, `list_turns_by_simulation()`, `list_turns_by_agent()` work
- [ ] `save_prediction()`, `get_prediction()`, `list_predictions()`, `update_prediction()` work
- [ ] All IDs use correct ULID prefixes (s_, ap_, st_, p_)
- [ ] All timestamps auto-populated as ISO 8601 UTC
- [ ] Edge cases: empty required fields raise ValueError

### P2.1: Agent Persona Generation
**Status:** NOT STARTED
**Depends on:** P2.0
**Create:** `forge/swarm/__init__.py`, `forge/swarm/population.py`, `forge/swarm/prompts.py`, `forge/swarm/prompts/persona_generator.md`, `tests/test_population.py`
**Acceptance criteria:**
- [ ] `SeedMaterial` dataclass with `text` and optional `context`
- [ ] `generate_population(seed, llm, count)` makes one LLM call, returns `list[AgentPersona]`
- [ ] Prompt template loaded from `persona_generator.md` with Jinja2
- [ ] Each persona has `ap_` ULID, archetype, full persona_json
- [ ] Handles empty agent list from LLM gracefully
- [ ] Handles partial/malformed agent data gracefully

### P2.3: Interaction Selection
**Status:** NOT STARTED
**Depends on:** P2.0, P2.1
**Create:** `forge/swarm/interaction.py`, `tests/test_interaction.py`
**Acceptance criteria:**
- [ ] `select_interactions()` returns 2-3 opposing views per agent (pure computation)
- [ ] Priority: strongest opposing, most contrarian, domain expert disagreement
- [ ] Parses persona_json for personality traits and expertise
- [ ] Never includes the agent's own turn
- [ ] Falls back gracefully when fewer opposing views exist
- [ ] Handles all-agree scenario (selects highest contrarian_tendency)

### P2.4: Consensus Extraction
**Status:** NOT STARTED
**Depends on:** P2.0
**Create:** `forge/swarm/consensus.py`, `tests/test_consensus.py`
**Acceptance criteria:**
- [ ] `ConsensusReport` dataclass with majority position, dissent clusters, conviction shifts, edge cases, prediction candidates
- [ ] `extract_consensus()` is pure computation (no LLM calls)
- [ ] Groups round 3 turns by position, identifies majority
- [ ] Tracks conviction shifts between round 1 and round 3
- [ ] Identifies edge cases (positions held by 1-2 agents)
- [ ] Handles ties (highest average confidence wins)
- [ ] Handles empty/single-agent input gracefully

### P2.2: Arena — Multi-Round Simulation Loop
**Status:** NOT STARTED
**Depends on:** P2.1, P2.3, P2.4
**Create:** `forge/swarm/arena.py`, `forge/swarm/prompts/reaction.md`, `forge/swarm/prompts/interaction.md`, `forge/swarm/prompts/convergence.md`, `tests/test_arena.py`
**Acceptance criteria:**
- [ ] `run_simulation()` runs 3 rounds with async batching via `asyncio.Semaphore`
- [ ] Round 1: independent reactions (parallel)
- [ ] Round 2: agents respond to 2-3 selected opposing views (uses P2.3)
- [ ] Round 3: final positions with conviction deltas
- [ ] Persists simulation record + all turns to store as it goes
- [ ] Updates simulation status (pending → running → complete/failed)
- [ ] Records duration_seconds
- [ ] Returns `SimulationResult` with simulation, turns, consensus

### P2.5: Prediction Extraction
**Status:** NOT STARTED
**Depends on:** P2.2, P2.4
**Create:** `forge/swarm/predictions.py`, `forge/swarm/prompts/extraction.md`, `tests/test_predictions.py`
**Acceptance criteria:**
- [ ] `extract_predictions()` makes one LLM call with consensus report
- [ ] Returns `list[Prediction]` persisted to store
- [ ] Updates `simulation.predictions_extracted` count
- [ ] Each prediction has claim, confidence, consensus_strength, resolution_deadline
- [ ] Handles empty predictions gracefully
- [ ] `format_consensus_for_prompt()` helper produces readable text

### P2.6: CLI `forge simulate` Command
**Status:** NOT STARTED
**Depends on:** P2.1–P2.5
**Modify:** `forge/cli.py`, `tests/test_cli.py`
**Acceptance criteria:**
- [ ] `forge simulate "scenario"` runs full pipeline and outputs Rich-formatted results
- [ ] `--context` / `-c` passes context
- [ ] `--agents N` / `-a` overrides swarm size (default from config: 30)
- [ ] `--rounds N` / `-r` overrides round count (default from config: 3)
- [ ] `--json` / `-j` outputs raw JSON
- [ ] Shows: simulation metadata, majority position, dissent, conviction shifts, predictions
- [ ] Non-zero exit on pipeline failure

### P2.7: Upgrade Path (Flag Only)
**Status:** NOT STARTED
**Depends on:** P2.6
**Modify:** `forge/cli.py`, `tests/test_cli.py`
**Acceptance criteria:**
- [ ] After `forge test` with confidence > 60, creates simulation record with status="queued"
- [ ] Prints suggestion: "Run `forge simulate` for deeper analysis"
- [ ] Low-confidence verdicts do NOT create queued simulations
- [ ] No scheduler, no auto-execution

### P2.8: Coverage + Cleanup
**Status:** NOT STARTED
**Depends on:** All above
**Acceptance criteria:**
- [ ] Coverage >= 85% on `forge/swarm/`
- [ ] New conftest.py fixtures: sample_simulation, sample_agent_persona, sample_seed
- [ ] All edge cases tested (0 agents, single agent, unanimous, empty)
- [ ] Total test count ~185+
