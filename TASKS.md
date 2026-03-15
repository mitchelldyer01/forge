# FORGE — Task Tracking

## Phase 0: The Kernel

### P0.1: Database layer (schema + store + models)

**Status:** IN PROGRESS

**Acceptance Criteria:**
- [ ] All tables from FORGE-ARCHITECTURE.md Section 4 created via schema.py
- [ ] All indexes from Section 4 created
- [ ] WAL mode enabled on file-backed connections
- [ ] In-memory DB works for tests
- [ ] ULID generation with type prefixes (h_, e_, r_, s_, st_, ap_, p_, a_, f_)
- [ ] All timestamps auto-populated as ISO 8601 UTC
- [ ] Pydantic models for: Hypothesis, Evidence, Relation, Simulation, AgentPersona, SimulationTurn, Prediction, Feed, Article, Feedback, CalibrationSnapshot
- [ ] Store provides typed CRUD methods:
  - Hypothesis: create, get_by_id, update, list_by_status, list_by_confidence_range
  - Evidence: create, get_by_id
  - Relation: create, list_by_source, list_by_target
  - Feedback: create
- [ ] All Store methods return Pydantic models, not raw dicts/tuples
- [ ] Edge cases handled: missing required fields raise errors, invalid status values rejected
