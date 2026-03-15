# FORGE — Calibrated Prediction Engine

## Architecture Specification v2.0

**Purpose:** This document is the complete technical specification for FORGE, a calibrated prediction engine that combines swarm-intelligence simulation with evolutionary hypothesis management and resolution tracking. It is written to be consumed by Claude Code for task decomposition and implementation.

**What makes FORGE different from MiroFish:** MiroFish simulates forward and produces a report. FORGE simulates forward, tracks whether predictions resolve as true or false, evolves its hypothesis population based on outcomes, and builds a compounding calibration record over time. The simulation is the generation mechanism. The calibration loop is the product.

**Repository:** `forge/`
**Language:** Python 3.12+
**License:** MIT
**Runtime:** Local-first on AMD Ryzen AI MAX+ 395, 128GB unified memory, Arch Linux (Omarchy), ROCm. Designed for eventual AWS migration.

---

## 1. System Overview

FORGE has two modes of operation that share a common knowledge graph and evolution engine:

### Mode 1: Claim Testing (interactive)
User submits a specific claim → agents simulate responses → verdict returned with confidence.

### Mode 2: Scenario Simulation (generative)
User provides seed material (news, policy, market signal) → system constructs a population of diverse agents → agents interact in a simulated environment → emergent predictions are extracted → predictions enter the hypothesis graph.

Both modes feed into the same pipeline:

```
        ┌─────────────────┐     ┌──────────────────┐
        │  CLAIM TESTING   │     │ SCENARIO SIMULATION│
        │  (interactive)   │     │   (generative)     │
        └────────┬─────────┘     └────────┬───────────┘
                 │                         │
                 ▼                         ▼
         ┌──────────────────────────────────────┐
         │         HYPOTHESIS GRAPH              │
         │  (claims, evidence, relations,        │
         │   confidence scores, lineage)         │
         └──────────────────┬───────────────────┘
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
         ┌──────────────┐     ┌──────────────┐
         │   EVOLUTION   │     │  CALIBRATION  │
         │  cull / fork  │     │   resolve +   │
         │  / mutate     │     │   score       │
         └──────┬────────┘     └──────┬────────┘
                │                      │
                └──────────┬───────────┘
                           ▼
                   ┌──────────────┐
                   │    OUTPUT     │
                   │ briefs, API,  │
                   │ track record  │
                   └──────────────┘
```

---

## 2. Core Concepts

### 2.1 The Swarm (Simulation Engine)

Inspired by MiroFish/OASIS, but designed for local inference. Instead of 5 structured debate agents, FORGE generates a **population of 20-100 diverse agents** per simulation. Each agent has:

- **Persona:** demographics, professional background, risk appetite, political lean, domain expertise, personality traits (generated per simulation from seed material)
- **Memory:** short-term (current simulation context) + long-term (injected from knowledge graph — what has FORGE learned from prior simulations?)
- **Behavioral rules:** encoded in the system prompt — how this persona type tends to reason, what biases they carry, what they pay attention to

Agents don't follow a rigid debate protocol. They **react to the scenario** and to each other's reactions over multiple rounds, producing emergent consensus, disagreement clusters, and edge-case perspectives that no structured debate would surface.

### 2.2 The Hypothesis Graph (Knowledge + Memory)

SQLite-backed graph of claims, evidence, and typed relationships. This is NOT a simulation artifact — it's the persistent, cross-simulation memory that makes FORGE compound over time. Every simulation's outputs feed into it. Every new simulation draws context from it.

### 2.3 The Calibration Loop (The Moat)

Every hypothesis with a time-bound prediction gets tracked for resolution. Over time, FORGE builds:
- Per-topic calibration curves ("when FORGE says 80% confidence on AI pricing claims, it's right 76% of the time")
- Per-agent-archetype accuracy scores ("contrarian personas produce better predictions on regulatory topics")
- Trend detection ("FORGE's accuracy on crypto predictions has degraded — the agent population may need rebalancing")

This is the thing nobody else is building. Simulation without accountability is entertainment. Simulation with calibration is intelligence.

### 2.4 Evolution Engine

Darwinian selection over the hypothesis population AND the agent population:
- **Hypothesis evolution:** weak claims die, strong claims fork into variants, contradictions get tested head-to-head
- **Agent evolution:** persona archetypes that consistently produce accurate predictions get reinforced; those that don't get culled and replaced. The agent population itself improves over time.

---

## 3. Project Structure

```
forge/
├── CLAUDE.md
├── BACKLOG.md
├── TASKS.md
├── README.md
├── pyproject.toml
├── forge/
│   ├── __init__.py
│   ├── cli.py                 # CLI entrypoint (Typer)
│   ├── config.py              # Settings via pydantic-settings + .env
│   ├── db/
│   │   ├── __init__.py
│   │   ├── schema.py          # SQLite schema + migrations
│   │   ├── store.py           # Repository pattern — all DB operations
│   │   └── models.py          # Pydantic models for all domain objects
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── rss.py             # RSS/Atom feed polling
│   │   ├── url.py             # URL content extraction (trafilatura)
│   │   └── manual.py          # CLI manual claim / seed material input
│   ├── extract/
│   │   ├── __init__.py
│   │   ├── claims.py          # Claim extraction from raw text
│   │   └── graph_build.py     # Entity/relationship extraction for seed material
│   ├── swarm/
│   │   ├── __init__.py
│   │   ├── population.py      # Agent persona generation from seed material
│   │   ├── agent.py           # Single agent: persona + memory + behavior
│   │   ├── arena.py           # Simulation environment: agents react + interact
│   │   ├── interaction.py     # Agent interaction types (react, reply, amplify, challenge)
│   │   ├── consensus.py       # Post-simulation: extract clusters, consensus, outliers
│   │   └── prompts/
│   │       ├── persona_generator.md   # Prompt to generate diverse agent personas
│   │       ├── reaction.md            # How agents react to a scenario
│   │       ├── interaction.md         # How agents respond to other agents
│   │       └── extraction.md          # Extract predictions from simulation output
│   ├── analyze/
│   │   ├── __init__.py
│   │   ├── structured.py      # Structured analysis mode (steelman/redteam/judge)
│   │   └── judge.py           # Synthesis: score claims from swarm or structured output
│   ├── evolve/
│   │   ├── __init__.py
│   │   ├── selection.py       # Hypothesis culling, promotion, forking
│   │   ├── mutation.py        # Hypothesis variant generation
│   │   ├── lineage.py         # Ancestry tracking
│   │   └── agent_evolution.py # Agent archetype performance tracking + evolution
│   ├── calibrate/
│   │   ├── __init__.py
│   │   ├── resolver.py        # Track prediction outcomes (true/false/partial)
│   │   ├── scorer.py          # Calibration curve computation
│   │   └── drift.py           # Detect calibration degradation over time
│   ├── retrieve/
│   │   ├── __init__.py
│   │   ├── embeddings.py      # Local embedding generation
│   │   └── search.py          # Similarity search over hypothesis graph
│   ├── export/
│   │   ├── __init__.py
│   │   ├── obsidian.py        # One-way Obsidian vault renderer
│   │   ├── digest.py          # Daily brief generator
│   │   └── api.py             # FastAPI server
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py          # Async HTTP client for llama.cpp
│   │   └── router.py          # Model routing (fast model vs reasoning model)
│   └── pipeline/
│       ├── __init__.py
│       ├── runner.py           # Main pipeline orchestration
│       └── scheduler.py        # Cron-like scheduling
├── tests/
│   ├── conftest.py
│   ├── test_db.py
│   ├── test_swarm.py
│   ├── test_analyze.py
│   ├── test_evolve.py
│   ├── test_calibrate.py
│   └── test_pipeline.py
├── scripts/
│   ├── setup_llama.sh
│   └── seed_feeds.py
└── data/
    ├── forge.db
    └── obsidian/
```

---

## 4. Data Model (SQLite)

Single file: `data/forge.db`. WAL mode for concurrent reads.

### 4.1 Core Tables

```sql
-- Hypotheses: the core unit of knowledge
CREATE TABLE hypotheses (
    id TEXT PRIMARY KEY,                -- h_{ulid}
    claim TEXT NOT NULL,                -- The hypothesis statement
    context TEXT,                       -- Background/framing
    confidence INTEGER DEFAULT 50,     -- 0-100
    status TEXT DEFAULT 'alive',       -- alive | dead | resolved_true | resolved_false | dormant
    resolution_deadline TEXT,          -- ISO 8601: when should this resolve by? (nullable)
    generation INTEGER DEFAULT 0,
    parent_id TEXT,
    source TEXT NOT NULL,               -- manual | rss | url | fork | mutation | simulation
    source_ref TEXT,                    -- URL, feed name, simulation ID, parent ID
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_challenged_at TEXT,
    challenges_survived INTEGER DEFAULT 0,
    challenges_failed INTEGER DEFAULT 0,
    human_endorsed INTEGER DEFAULT 0,
    human_rejected INTEGER DEFAULT 0,
    tags TEXT,                          -- JSON array
    FOREIGN KEY (parent_id) REFERENCES hypotheses(id)
);

-- Evidence: facts that support or contradict hypotheses
CREATE TABLE evidence (
    id TEXT PRIMARY KEY,                -- e_{ulid}
    content TEXT NOT NULL,
    source_url TEXT,
    source_name TEXT,
    published_at TEXT,
    ingested_at TEXT NOT NULL,
    embedding BLOB
);

-- Typed relationships
CREATE TABLE relations (
    id TEXT PRIMARY KEY,                -- r_{ulid}
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,        -- supports | contradicts | refines | evidence_for | evidence_against | supersedes
    strength REAL DEFAULT 0.5,         -- 0.0-1.0
    reasoning TEXT,
    created_at TEXT NOT NULL,
    source_simulation_id TEXT,         -- which simulation established this
    FOREIGN KEY (source_simulation_id) REFERENCES simulations(id)
);

-- Simulations: a complete simulation run
CREATE TABLE simulations (
    id TEXT PRIMARY KEY,                -- s_{ulid}
    mode TEXT NOT NULL,                 -- claim_test | scenario
    seed_text TEXT NOT NULL,            -- The input claim or seed material
    seed_context TEXT,
    agent_count INTEGER,
    rounds INTEGER,
    status TEXT DEFAULT 'pending',     -- pending | running | complete | failed
    summary TEXT,                       -- Post-simulation synthesis
    predictions_extracted INTEGER DEFAULT 0,
    started_at TEXT,
    completed_at TEXT,
    duration_seconds REAL
);

-- Agent personas generated for simulations
CREATE TABLE agent_personas (
    id TEXT PRIMARY KEY,                -- ap_{ulid}
    archetype TEXT NOT NULL,            -- e.g., "retail_investor", "policy_analyst", "tech_optimist"
    persona_json TEXT NOT NULL,         -- Full persona definition (JSON)
    simulations_participated INTEGER DEFAULT 0,
    predictions_correct INTEGER DEFAULT 0,
    predictions_incorrect INTEGER DEFAULT 0,
    calibration_score REAL,            -- Running accuracy metric
    active INTEGER DEFAULT 1,          -- Evolved out = 0
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Individual agent reactions within a simulation
CREATE TABLE simulation_turns (
    id TEXT PRIMARY KEY,                -- st_{ulid}
    simulation_id TEXT NOT NULL,
    round INTEGER NOT NULL,
    agent_persona_id TEXT NOT NULL,
    turn_type TEXT NOT NULL,            -- reaction | reply | challenge | amplify | consensus_shift
    content TEXT NOT NULL,              -- Agent's output
    responding_to_id TEXT,             -- st_ ID this is responding to (nullable for initial reactions)
    position TEXT,                      -- support | oppose | conditional | neutral
    confidence INTEGER,                -- Agent's confidence in their position
    token_count INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (simulation_id) REFERENCES simulations(id),
    FOREIGN KEY (agent_persona_id) REFERENCES agent_personas(id)
);

-- Extracted predictions from simulations (become hypotheses)
CREATE TABLE predictions (
    id TEXT PRIMARY KEY,                -- p_{ulid}
    simulation_id TEXT NOT NULL,
    hypothesis_id TEXT,                 -- FK once converted to hypothesis
    claim TEXT NOT NULL,
    confidence INTEGER NOT NULL,
    consensus_strength REAL,           -- How much of the swarm agreed (0.0-1.0)
    dissent_summary TEXT,              -- Summary of opposing views
    resolution_deadline TEXT,          -- When this should be checkable
    resolved_at TEXT,
    resolved_as TEXT,                  -- true | false | partial | unresolvable
    resolution_evidence TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (simulation_id) REFERENCES simulations(id),
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(id)
);

-- RSS/feed sources
CREATE TABLE feeds (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    url TEXT NOT NULL UNIQUE,
    feed_type TEXT DEFAULT 'rss',
    active INTEGER DEFAULT 1,
    last_polled_at TEXT,
    poll_interval_minutes INTEGER DEFAULT 240
);

-- Ingested articles
CREATE TABLE articles (
    id TEXT PRIMARY KEY,                -- a_{ulid}
    feed_id TEXT,
    url TEXT UNIQUE,
    title TEXT,
    content TEXT,
    published_at TEXT,
    ingested_at TEXT NOT NULL,
    claims_extracted INTEGER DEFAULT 0,
    FOREIGN KEY (feed_id) REFERENCES feeds(id)
);

-- Human feedback
CREATE TABLE feedback (
    id TEXT PRIMARY KEY,
    hypothesis_id TEXT,
    prediction_id TEXT,
    action TEXT NOT NULL,               -- endorse | reject | resolve_true | resolve_false | resolve_partial | annotate
    note TEXT,
    created_at TEXT NOT NULL
);

-- Calibration snapshots (computed periodically)
CREATE TABLE calibration_snapshots (
    id TEXT PRIMARY KEY,
    computed_at TEXT NOT NULL,
    total_predictions INTEGER,
    resolved_predictions INTEGER,
    accuracy_overall REAL,             -- % correct of resolved
    calibration_json TEXT,             -- JSON: {bucket: "70-80", predicted: N, correct: N, accuracy: %}
    topic_breakdown_json TEXT,         -- JSON: {topic: {accuracy, count}}
    archetype_breakdown_json TEXT      -- JSON: {archetype: {accuracy, count}}
);

-- Indexes
CREATE INDEX idx_hypotheses_status ON hypotheses(status);
CREATE INDEX idx_hypotheses_confidence ON hypotheses(confidence);
CREATE INDEX idx_hypotheses_tags ON hypotheses(tags);
CREATE INDEX idx_relations_source ON relations(source_id);
CREATE INDEX idx_relations_target ON relations(target_id);
CREATE INDEX idx_simulations_status ON simulations(status);
CREATE INDEX idx_simulation_turns_sim ON simulation_turns(simulation_id);
CREATE INDEX idx_predictions_simulation ON predictions(simulation_id);
CREATE INDEX idx_predictions_resolved ON predictions(resolved_as);
CREATE INDEX idx_agent_personas_archetype ON agent_personas(archetype);
CREATE INDEX idx_articles_url ON articles(url);
```

### 4.2 ID Generation

ULIDs with type prefixes: `h_`, `e_`, `r_`, `s_`, `st_`, `ap_`, `p_`, `a_`, `f_`. Package: `python-ulid`.

### 4.3 Embedding Storage

Store as BLOB (numpy float32 via `tobytes()`). Brute-force cosine similarity in numpy for < 10K entries. Migrate to `sqlite-vec` at scale.

---

## 5. Inference Backend

### 5.1 llama.cpp Server

```bash
llama-server \
  -m models/qwen3-8b-q4_k_m.gguf \
  -c 32768 \
  -np 8 \
  -cb \
  --host 127.0.0.1 \
  --port 8080 \
  -ngl 99 \
  --jinja \
  --metrics
```

- 8 parallel slots, continuous batching, full GPU offload via ROCm.
- 4096 tokens per slot (32768 / 8).
- All agents hit this single endpoint via OpenAI-compatible API.

### 5.2 Model Routing (Phase 3+)

Run two llama-server instances:

| Instance | Port | Model | Use |
|----------|------|-------|-----|
| Primary | 8080 | Qwen3-8B Q4_K_M | Swarm reactions, judge synthesis, complex reasoning |
| Fast | 8081 | Qwen3-4B Q4_K_M or Qwen3-1.7B | Claim extraction, persona generation, relevance filtering, embedding |

The `forge/llm/router.py` selects the endpoint based on task type.

### 5.3 Client Interface (`forge/llm/client.py`)

```python
class LLMClient:
    async def complete(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: dict | None = None,
    ) -> CompletionResponse: ...
```

- Retry with exponential backoff on 503.
- JSON mode for all structured outputs.
- Configurable timeouts per task type.
- Token counting logged per call.

---

## 6. Swarm Simulation Engine

This is the core generation mechanism. It replaces the rigid 5-agent debate from v1.

### 6.1 Population Generation (`forge/swarm/population.py`)

Given seed material, generate a diverse agent population:

```python
async def generate_population(
    seed: SeedMaterial,
    count: int = 30,        # Default swarm size (tunable)
    diversity_axes: list[str] | None = None,
) -> list[AgentPersona]:
```

**Persona generation prompt:**

```
Given the following scenario, generate {count} diverse agent personas who 
would have meaningful, distinct perspectives on this topic.

SCENARIO: {seed.text}

For each agent, output JSON:
{
  "agents": [
    {
      "archetype": "short label (e.g., retail_investor, policy_hawk, tech_founder)",
      "name": "realistic name",
      "background": "1-2 sentence professional/personal background",
      "expertise": ["domain1", "domain2"],
      "personality": {
        "risk_appetite": "low|medium|high",
        "optimism_bias": "pessimist|realist|optimist",
        "contrarian_tendency": 0.0-1.0,
        "analytical_depth": "surface|moderate|deep"
      },
      "initial_stance": "1 sentence gut reaction to the scenario",
      "reasoning_style": "how this person tends to think about problems"
    }
  ]
}

DIVERSITY REQUIREMENTS:
- At least 20% should be contrarian (disagree with the likely majority view).
- Include at least 2 domain experts directly relevant to the scenario.
- Include at least 2 "adjacent domain" experts who bring unexpected perspectives.
- Include at least 1 pessimist, 1 optimist, and 1 who focuses purely on second-order effects.
- Vary risk appetites, analytical depth, and reasoning styles across the population.
- No two agents should have the same archetype + stance combination.
```

**Persona reuse:** Successful personas (high calibration scores) persist across simulations in the `agent_personas` table. New simulations draw from existing high-performing personas + generate fresh ones for diversity. This is how the agent population evolves.

### 6.2 Arena (`forge/swarm/arena.py`)

The arena manages the multi-round simulation loop.

```python
async def run_simulation(
    seed: SeedMaterial,
    population: list[AgentPersona],
    rounds: int = 3,
    injections: list[Injection] | None = None,  # "God's eye" variable injection
) -> SimulationResult:
```

**Round structure:**

```
Round 1 — Initial Reactions:
  Each agent independently reacts to the seed material.
  Output: position (support/oppose/conditional/neutral), confidence, reasoning.
  All agents fire in parallel (batched through llama.cpp slots).

Round 2 — Interaction:
  Each agent reads a curated sample of Round 1 reactions (not all — simulates
  information asymmetry). Each agent responds to 2-3 other agents they disagree with.
  Agents can: challenge, amplify, refine, or shift position.
  
  [OPTIONAL] Variable injection: user can inject a new variable here
  ("Now assume the Fed cuts rates by 50bp") and agents react to the changed scenario.

Round 3 — Convergence:
  Each agent gives a final position accounting for the full debate.
  Agents explicitly state whether they changed their mind and why.
  Output includes a "conviction delta" — how much their confidence shifted.
```

**Interaction selection:** In Round 2, each agent doesn't see ALL other reactions (that's unrealistic and expensive). The arena selects 2-3 opposing views for each agent to respond to, prioritizing:
1. Strongest opposing argument (highest confidence from the other side)
2. Most unexpected perspective (highest contrarian_tendency agent on the other side)
3. Domain expert disagreement (agent with relevant expertise who disagrees)

This produces more productive interactions than random pairing.

**Concurrency:** With 30 agents and 8 llama.cpp slots:
- Round 1: 30 calls, 8 at a time = ~4 batches
- Round 2: 30 calls (each agent responds to 2-3 others, but we batch the calls) = ~4 batches
- Round 3: 30 calls = ~4 batches
- Total: ~12 batches × ~10s per batch = ~2 minutes per simulation

With 100 agents: ~13 batches per round × 3 rounds = ~39 batches = ~7 minutes. Acceptable.

### 6.3 Consensus Extraction (`forge/swarm/consensus.py`)

Post-simulation analysis. No LLM calls — this is pure computation on structured output.

```python
def extract_consensus(turns: list[SimulationTurn]) -> ConsensusReport:
```

Produces:
- **Majority position:** what most agents ended up believing, with confidence distribution
- **Dissent clusters:** groups of agents who disagree with majority, clustered by reasoning similarity
- **Conviction shifts:** which agents changed their minds and what caused it (strongest signal)
- **Edge cases:** unique perspectives held by only 1-2 agents (often the most interesting)
- **Predictions:** specific, time-bound, falsifiable claims extracted from the simulation

The predictions are the key output. Each gets a confidence score derived from:
- % of swarm that supports it
- Average confidence of supporting agents
- Whether domain experts specifically support it
- Whether any agent who shifted TO this position did so (shifts are higher signal than initial stances)

### 6.4 Prediction Extraction Prompt

After computational consensus analysis, one LLM call extracts explicit predictions:

```
You are analyzing the output of a multi-agent simulation about:
{seed.text}

SIMULATION SUMMARY:
{consensus_report}

Extract specific, falsifiable predictions from this simulation. Each prediction must:
1. Be a concrete claim about what will happen
2. Have a timeframe (when could this be verified?)
3. Have clear resolution criteria (how do we know if it's true or false?)

Output JSON:
{
  "predictions": [
    {
      "claim": "specific prediction",
      "confidence": 0-100,
      "consensus_strength": 0.0-1.0,
      "resolution_deadline": "ISO 8601 date",
      "resolution_criteria": "how to verify this",
      "dissent_summary": "what the minority argued",
      "tags": ["topic1", "topic2"]
    }
  ]
}
```

---

## 7. Structured Analysis Mode (Claim Testing)

For interactive `forge test "claim"`, use a lighter-weight structured analysis instead of a full swarm simulation. This is faster (9 LLM calls vs 90+) and appropriate for quick hypothesis testing.

### 7.1 Three-Pass Analysis

```
Pass 1 — Steelman: Best case for the claim.
Pass 2 — Redteam: Strongest attack on the claim.
Pass 3 — Judge: Synthesize, assign confidence, identify conditions and relations.
```

All three use JSON output format. The Judge also identifies relations to existing hypotheses in the graph.

### 7.2 Upgrade Path

If a claim tested via structured analysis scores confidence > 60 OR is tagged as high-importance by the user, it gets automatically queued for a full swarm simulation. This creates a natural funnel: quick test → full simulation for claims that warrant it.

---

## 8. Calibration Engine

### 8.1 Resolution Tracking (`forge/calibrate/resolver.py`)

Every prediction with a `resolution_deadline` gets checked:

```python
async def check_resolutions():
    pending = store.get_predictions_past_deadline()
    for prediction in pending:
        # Option 1: Automated (search for resolution evidence)
        evidence = await search_for_resolution(prediction)
        if evidence.is_conclusive:
            store.resolve_prediction(prediction.id, evidence.outcome, evidence.text)
        # Option 2: Queue for human review
        else:
            store.flag_for_review(prediction.id)
```

Automated resolution uses web search (in Phase 5+) or relies on newly ingested evidence that contradicts or confirms the prediction. In early phases, resolution is primarily human-driven via `forge resolve p_xxxxx --true/--false`.

### 8.2 Calibration Scoring (`forge/calibrate/scorer.py`)

Compute calibration curves by bucketing predictions:

```python
def compute_calibration(predictions: list[ResolvedPrediction]) -> CalibrationReport:
    buckets = {}  # {(70,80): {"total": 10, "correct": 7}}
    for p in predictions:
        bucket = (p.confidence // 10 * 10, p.confidence // 10 * 10 + 10)
        buckets[bucket]["total"] += 1
        if p.resolved_as == "true":
            buckets[bucket]["correct"] += 1
    # Perfect calibration: 70-80% bucket should be ~75% accurate
    return CalibrationReport(buckets=buckets, brier_score=compute_brier(predictions))
```

Also compute:
- **Brier score** (proper scoring rule for probabilistic predictions)
- **Per-topic accuracy** (are we better at AI predictions than geopolitical ones?)
- **Per-archetype accuracy** (which agent personas produce better predictions?)
- **Temporal accuracy** (are we getting better or worse over time?)

### 8.3 Drift Detection (`forge/calibrate/drift.py`)

Alert when:
- Rolling 30-day Brier score degrades by > 15% vs the prior 30 days
- A specific topic's accuracy drops below 50% (worse than a coin flip)
- A high-performing archetype's accuracy suddenly drops (may indicate a domain shift)

Drift detection triggers: re-evaluation of agent population composition, potential prompt tuning, or flagging to the user that FORGE's predictions in a domain should be trusted less.

---

## 9. Evolution Engine

### 9.1 Hypothesis Evolution (`forge/evolve/selection.py`)

Runs daily.

**Culling:**
- `confidence < 25` AND `challenges_survived < 2` AND `age > 7 days` → dead
- `human_rejected >= 3` AND `human_endorsed == 0` → dead
- No activity in 30 days → dormant
- `resolved_false` → dead (with lineage preserved)

**Promotion:**
- `confidence >= 75` AND `challenges_survived >= 3` → tag 'high_conviction'
- `human_endorsed >= 2` → confidence += 10 (cap 95)
- `resolved_true` → tag 'confirmed', becomes anchor evidence for related hypotheses

**Forking:**
- Top 10% by confidence → generate 1-2 variants:
  - Stronger: tighten the claim, add specificity
  - Adjacent: apply same logic to different domain
  - Temporal: "this will happen sooner/later than expected"

**Re-challenge:**
- Hypotheses not debated in 14+ days with new relevant evidence → auto-queue for swarm simulation.

### 9.2 Agent Population Evolution (`forge/evolve/agent_evolution.py`)

Runs weekly.

**Tracking:** Each `agent_persona` accumulates:
- `simulations_participated`: how many simulations they've been in
- `predictions_correct` / `predictions_incorrect`: from resolved predictions where they were on the majority/minority side
- `calibration_score`: rolling accuracy

**Selection:**
- Personas with `simulations_participated >= 5` AND `calibration_score < 0.4` → deactivated
- Personas with `calibration_score > 0.7` → preferred for future simulations (drawn first from pool)
- When the active persona pool drops below threshold → generate new personas to replenish diversity

**Mutation:**
- Top-performing personas get "variant" personas generated: same background but different risk appetite, or same expertise but contrarian tendency flipped. Test whether the variant outperforms.

This means **the swarm itself evolves over time** — it's not just the hypotheses getting better, it's the agents producing them.

---

## 10. Ingestion Pipeline

### 10.1 Manual Input (CLI)

```bash
# Quick claim test (structured analysis, fast)
forge test "Outcome-contingent pricing will outperform flat-rate SaaS"

# Full swarm simulation from claim
forge simulate "What happens if the EU regulates AI agents as financial advisors?"

# Simulate from URL (extracts seed material, runs swarm)
forge simulate --url https://example.com/article

# Simulate with variable injection
forge simulate "Fed announces 50bp rate cut" --inject "China retaliates with tariffs"

# Feedback
forge endorse h_00483
forge reject h_00291
forge resolve p_00123 --true --note "Confirmed by Q2 earnings report"

# Query
forge ask "What do I believe about AI agent pricing?"
forge calibration                    # Show calibration report
forge calibration --topic "crypto"   # Topic-specific calibration
forge leaderboard                    # Top performing agent archetypes
forge brief                          # Generate today's digest
forge status                         # Pipeline health + stats
```

### 10.2 RSS Ingestion (Continuous)

Same as v1 spec. Polls feeds, extracts content, extracts claims, filters by relevance, queues for simulation.

### 10.3 Seed Material Processing (`forge/extract/graph_build.py`)

For scenario simulations, seed material gets pre-processed:

1. **Entity extraction:** identify key actors, organizations, concepts
2. **Relationship mapping:** how entities relate to each other
3. **Context injection:** pull relevant existing hypotheses from the graph
4. **Scenario framing:** generate a clear scenario statement for agents to react to

This is a single LLM call with structured output. The result is injected into each agent's context.

---

## 11. Output Layer

### 11.1 Obsidian Export

One-way render from SQLite. Same structure as v1 but with simulation records added.

### 11.2 Daily Digest

Content:
1. **New predictions** generated in last 24h (from simulations)
2. **Resolved predictions** — what came true, what didn't
3. **Calibration update** — running accuracy, any drift alerts
4. **High-conviction hypotheses** — recently promoted
5. **Killed hypotheses** — recently died and why
6. **Active contradictions** — pairs that contradict each other
7. **Agent leaderboard** — top 5 performing archetypes

### 11.3 API Server (FastAPI)

```
POST   /v1/test              # Quick claim test → structured analysis verdict
POST   /v1/simulate           # Full swarm simulation → predictions
GET    /v1/hypotheses          # List/filter hypotheses
GET    /v1/hypotheses/{id}     # Hypothesis with relations
GET    /v1/predictions         # List predictions with resolution status
POST   /v1/feedback            # Human feedback
GET    /v1/calibration         # Calibration report
GET    /v1/calibration/{topic} # Topic-specific calibration
GET    /v1/leaderboard         # Agent archetype performance
GET    /v1/brief               # Latest daily digest
GET    /v1/ask                 # Natural language query
GET    /v1/health
GET    /v1/stats
```

### 11.4 Track Record Page (Revenue Surface)

A public-facing page showing FORGE's prediction track record:
- Total predictions made, resolved, accuracy
- Calibration curve visualization
- Recent correct and incorrect predictions
- Topic-specific accuracy

This is the trust artifact. It's what makes someone pay for FORGE's analysis — demonstrable, auditable accuracy over time.

---

## 12. Configuration

```bash
# .env
FORGE_DB_PATH=data/forge.db
FORGE_LLAMA_URL=http://127.0.0.1:8080
FORGE_LLAMA_FAST_URL=http://127.0.0.1:8081     # Optional fast model
FORGE_EMBED_MODEL=all-MiniLM-L6-v2
FORGE_DEFAULT_SWARM_SIZE=30
FORGE_SIMULATION_ROUNDS=3
FORGE_PARALLEL_SIMULATIONS=1                     # Max concurrent simulations
FORGE_RELEVANCE_THRESHOLD=0.6
FORGE_CULL_MIN_CONFIDENCE=25
FORGE_CULL_MIN_AGE_DAYS=7
FORGE_RECHALLENGE_DAYS=14
FORGE_CALIBRATION_SNAPSHOT_DAYS=7                # Compute calibration weekly
FORGE_OBSIDIAN_PATH=data/obsidian
FORGE_DIGEST_EMAIL=
FORGE_LOG_LEVEL=INFO
```

---

## 13. Dependencies

```toml
[project]
name = "forge"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "typer>=0.15",
    "httpx>=0.28",
    "pydantic>=2.10",
    "pydantic-settings>=2",
    "python-ulid>=3",
    "trafilatura>=2",
    "feedparser>=6",
    "sentence-transformers>=4",
    "numpy>=2",
    "fastapi>=0.115",
    "uvicorn>=0.34",
    "rich>=13",
    "jinja2>=3",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.25",
    "ruff>=0.9",
]

[project.scripts]
forge = "forge.cli:app"
```

---

## 14. Testing Strategy

### 14.1 Unit Tests
- `test_db.py`: Schema, CRUD, relation queries, embedding storage.
- `test_swarm.py`: Population generation parsing, consensus extraction (pure computation, no LLM). Arena round management with mock agents.
- `test_calibrate.py`: Calibration curve computation, Brier score, drift detection with fixture data.
- `test_evolve.py`: Selection logic with fixture data.

### 14.2 Integration Tests
- `test_analyze.py`: Structured analysis flow with mock LLM.
- `test_pipeline.py`: End-to-end from manual claim through analysis to hypothesis storage.

### 14.3 Mock LLM Client
`MockLLMClient` returns deterministic JSON responses. Tests never hit llama.cpp.

### 14.4 Quality Tests (Manual)
After each phase: 10 real claims, manual evaluation. Additionally, after Phase 3: run 5 scenario simulations, evaluate whether swarm output surfaces insights that structured analysis missed.

---

## 15. CLAUDE.md

```markdown
# FORGE — Calibrated Prediction Engine

## Commands
- `uv run pytest` — run all tests
- `uv run ruff check forge/` — lint
- `uv run forge test "claim"` — quick structured analysis
- `uv run forge simulate "scenario"` — full swarm simulation
- `uv run forge status` — system health
- `uv run forge calibration` — prediction track record

## Architecture
- See FORGE-ARCHITECTURE.md for full spec (v2.0)
- Two modes: structured analysis (fast, 3 LLM calls) and swarm simulation (thorough, 90+ calls)
- SQLite (WAL mode) is the single source of truth
- All LLM calls go through forge/llm/client.py → llama.cpp OpenAI API
- Agents are personas (JSON) + prompts (markdown), not code
- Pipeline: INGEST → EXTRACT → SIMULATE → JUDGE → EVOLVE → CALIBRATE

## Key Concept
The simulation generates predictions. The calibration loop tracks whether
predictions come true. The evolution engine kills bad hypotheses and bad
agent archetypes, and promotes good ones. Over time, FORGE gets more accurate.
This is the product. The simulation is just the generation mechanism.

## Rules
- No ORM. Raw SQL via store.py with Pydantic models.
- All IDs are ULIDs with type prefixes (h_, e_, r_, s_, st_, ap_, p_, a_, f_).
- All timestamps are ISO 8601 UTC.
- All LLM responses must be parsed as JSON. Handle malformed JSON gracefully.
- Never call the LLM directly. Always go through LLMClient.
- Tests use MockLLMClient. Never hit real inference in tests.
- Keep prompts in markdown files, not hardcoded strings.
- One function, one job. Small files. Explicit > implicit.
- Swarm size is configurable. Default 30. Can scale to 100+ but test throughput first.
```

---

## 16. Implementation Phases

### Phase 0: The Kernel (Week 1-2)
**Scope:** `forge test "claim"` → steelman → redteam → judge → JSON verdict + stored in DB.
**Files:** cli.py, config.py, db/, llm/client.py, analyze/structured.py, analyze/judge.py, basic prompts.
**No:** Swarm, evolution, calibration, RSS, Obsidian, API, embeddings.
**Validation:** 10 real claims, 7+ produce useful analysis.

### Phase 1: Memory & Retrieval (Week 3-4)
**Scope:** Hypotheses persist. Retrieval injects prior context into analysis. Relations tracked.
**Adds:** retrieve/, relation extraction in judge, `forge history`, `forge graph`.
**Validation:** 50th claim is demonstrably better-analyzed than 1st.

### Phase 2: Swarm Simulation (Week 5-8)
**Scope:** `forge simulate "scenario"` → population generation → multi-round arena → consensus extraction → predictions stored.
**Adds:** swarm/ (population, agent, arena, interaction, consensus), prediction extraction.
**Validation:** 5 scenario simulations produce predictions that structured analysis alone would not have surfaced. Emergent insights visible in output.

### Phase 3: Continuous Ingestion + Calibration (Week 9-12)
**Scope:** RSS feeds, claim extraction, relevance filtering, scheduled pipeline, resolution tracking, calibration scoring.
**Adds:** ingest/, extract/, calibrate/, pipeline/, feed management CLI.
**Validation:** Runs unattended 48h. At least 10 predictions have resolution deadlines. Calibration report computes correctly.

### Phase 4: Evolution & Output (Week 13-16)
**Scope:** Darwinian selection (hypotheses + agents), Obsidian export, daily digest, feedback loop, API, agent leaderboard.
**Adds:** evolve/, export/, feedback commands, daily cron, track record page.
**Validation:** Used daily for 2 weeks. Agent population has measurably evolved. At least 5 predictions resolved with calibration data.

### Phase 5: Hardening + Scale (Week 17-20)
**Scope:** Model routing (dual llama-server), monitoring, automated resolution attempts, backup, performance tuning.
**Adds:** llm/router.py, health checks, metrics, automated resolution via web search, backup scripts.
**Validation:** 4+ weeks unattended. Calibration stable. Agent evolution producing measurable accuracy improvement.

### Phase 6: AWS Migration + Revenue (Week 21+)
**Scope:** Cloud deployment, multi-tenant API, public track record, monetization.
**Infra:** Step Functions orchestration, DynamoDB hypothesis store, SQS job queue, Bedrock inference, API Gateway.
**Revenue:** API access (per-prediction pricing), paid daily briefs, self-service platform.

---

## 17. Key Design Decisions (Do Not Revisit)

1. **SQLite, not Postgres/Neo4j.** Single file, zero ops, WAL concurrency. Migrate at Phase 6.
2. **llama.cpp, not Ollama.** Direct slot control, no tool-calling bugs, no abstraction tax.
3. **Agents are personas + prompts, not code.** Swap by editing JSON/markdown.
4. **JSON mode for all LLM outputs.** Structured data everywhere. No prose parsing.
5. **ULIDs, not UUIDs.** Sortable by creation time.
6. **Two modes (structured + swarm) not one.** Quick test for interactive use, full simulation for deep analysis. Both feed the same graph.
7. **Calibration is a first-class feature, not an afterthought.** Every prediction gets a resolution deadline. Track record is the product.
8. **Agent population evolves.** Personas are persistent, tracked, and selected for. The swarm improves over time.
9. **Obsidian is read-only export.** One-way data flow. SQLite is truth.
10. **Pydantic for all data boundaries.** LLM responses, DB rows, API — everything validates.

---

## 18. What FORGE Is Not

- **Not MiroFish.** MiroFish simulates social media platforms (Twitter/Reddit clones). FORGE simulates diverse expert perspectives reacting to scenarios. Different simulation topology, same swarm insight.
- **Not a chat interface.** FORGE is a pipeline that produces artifacts (predictions, briefs, calibration reports). Interactive querying exists but is secondary to the autonomous loop.
- **Not a general-purpose agent framework.** FORGE does one thing: generate, test, and calibrate predictions. The agent infrastructure serves that purpose only.
- **Not trying to predict specific prices or outcomes.** FORGE predicts directional trends and conditional outcomes ("if X then likely Y"). Calibration tracks accuracy of these probabilistic claims, not point predictions.
