# FORGE — Calibrated Prediction Engine

A local-first prediction engine that generates, tests, and calibrates predictions using swarm-intelligence simulation. Runs on AMD Ryzen AI MAX+ 395 with Qwen3-8B via llama.cpp.

FORGE simulates diverse expert perspectives reacting to scenarios, tracks whether its predictions come true, and evolves both its hypotheses and its agent population over time. The simulation generates predictions. The calibration loop is the product.

## Prerequisites

- **Hardware:** AMD Ryzen AI MAX+ 395 (or any ROCm-capable GPU — adjust `AMDGPU_TARGETS` in `Dockerfile.llama`)
- **OS:** Arch Linux (tested on Omarchy) with ROCm kernel support
- **Software:**
  - Python 3.12+
  - [uv](https://docs.astral.sh/uv/) (Python package manager)
  - Docker (for llama.cpp with ROCm)
  - [huggingface-cli](https://huggingface.co/docs/huggingface_hub/guides/cli) (`uv tool install huggingface-hub[hf_xet]`)

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/mitchelldyer01/forge.git
cd forge
uv sync --all-extras
```

### 2. Download the model and build the inference server

```bash
bash scripts/setup_llama.sh
```

This downloads Qwen3-8B-Q4_K_M (~5 GB) from Hugging Face and builds the `forge-llama` Docker image with ROCm support for gfx1151.

### 3. Start llama-server

```bash
docker run --rm -d \
  --name forge-llama \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  -v ./models:/models \
  -p 8080:8080 \
  forge-llama
```

This starts Qwen3-8B with 8 parallel slots, continuous batching, full GPU offload, and 32K context.

### 4. Validate the server

```bash
python scripts/validate_model.py
# Add -v for verbose output with full model responses
python scripts/validate_model.py -v
```

All 9 checks should pass: health, model loaded, basic completion, thinking mode, JSON output, structured output, concurrent requests (8 slots), throughput, and JSON reliability (10/10).

### 5. Initialize the database

```bash
mkdir -p data
uv run forge status
```

The database (`data/forge.db`) is created automatically on first use.

### 6. Test a claim

```bash
uv run forge test "Outcome-contingent pricing will outperform flat-rate SaaS by 2027"
```

This runs a 3-pass structured analysis (steelman, redteam, judge) and stores the hypothesis in the database.

## Usage

### Interactive Analysis

```bash
# Quick claim test (3 LLM calls, ~30s)
uv run forge test "The EU will regulate AI agents as financial advisors by 2026"

# With context
uv run forge test "Bitcoin will hit 200k" --context "Post-halving cycle, ETF inflows"

# JSON output for scripting
uv run forge test "claim" --json
```

### Swarm Simulation

```bash
# Full swarm simulation (30 agents, 3 rounds, 90+ LLM calls, ~2 min)
uv run forge simulate "What happens if the EU regulates AI agents as financial advisors?"

# Custom agent count and rounds
uv run forge simulate "Fed cuts rates by 50bp" --agents 50 --rounds 4

# JSON output
uv run forge simulate "scenario" --json
```

### Prediction Tracking

```bash
# List all predictions
uv run forge predictions

# List overdue predictions
uv run forge predictions --overdue

# Resolve a prediction
uv run forge resolve p_XXXXX --true --note "Confirmed by Q2 earnings report"
uv run forge resolve p_XXXXX --false
uv run forge resolve p_XXXXX --partial
```

### Calibration

```bash
# View calibration report (accuracy, Brier score, bucket breakdown)
uv run forge calibration

# Agent performance leaderboard
uv run forge leaderboard
```

### Continuous Pipeline (RSS Ingestion)

```bash
# Add feeds
uv run forge feed add https://feeds.example.com/rss --name "Example Feed"

# List feeds
uv run forge feed list

# Poll feeds manually
uv run forge feed poll

# Run continuous pipeline (polls feeds, extracts claims, flags overdue predictions)
uv run forge run

# Single pipeline cycle
uv run forge run --once
```

### Evolution & Maintenance

```bash
# Run hypothesis + agent evolution cycle
uv run forge evolve

# Human feedback
uv run forge endorse h_XXXXX --note "Strong signal from earnings data"
uv run forge reject h_XXXXX

# Daily digest
uv run forge brief

# Export to Obsidian vault
uv run forge export ~/vault/forge

# View hypothesis graph
uv run forge graph h_XXXXX

# Browsing history
uv run forge history
uv run forge history --status alive
```

### System Health

```bash
uv run forge status
```

## Configuration

All settings are configurable via environment variables with the `FORGE_` prefix. Create a `.env` file in the project root:

```bash
# .env
FORGE_DB_PATH=data/forge.db
FORGE_LLAMA_URL=http://127.0.0.1:8080
FORGE_LLAMA_TIMEOUT=120.0
FORGE_DEFAULT_SWARM_SIZE=30
FORGE_SIMULATION_ROUNDS=3
FORGE_RELEVANCE_THRESHOLD=0.6
FORGE_POLL_INTERVAL_MINUTES=240
FORGE_CALIBRATION_SNAPSHOT_DAYS=7
FORGE_CULL_MIN_CONFIDENCE=25
FORGE_CULL_MIN_AGE_DAYS=7
FORGE_RECHALLENGE_DAYS=14
FORGE_LOG_LEVEL=INFO
```

## Building Data

To start building a calibration track record immediately:

1. **Seed with manual claims.** Test 10-20 claims you have opinions about:
   ```bash
   uv run forge test "AI coding assistants will handle 50% of production code by 2027"
   uv run forge test "The Fed will cut rates at least 3 times in 2026"
   uv run forge test "Apple will ship an AI agent platform in iOS 20"
   ```

2. **Run simulations on complex scenarios.** These generate multiple time-bound predictions:
   ```bash
   uv run forge simulate "Impact of EU AI Act enforcement on US AI startups"
   uv run forge simulate "What happens to SaaS pricing if AI agents replace human users?"
   ```

3. **Add RSS feeds for continuous ingestion.** The pipeline extracts claims automatically:
   ```bash
   uv run forge feed add https://feeds.arstechnica.com/arstechnica/technology-lab --name "Ars Technica"
   uv run forge feed add https://www.techmeme.com/feed.xml --name "Techmeme"
   uv run forge run  # starts continuous polling
   ```

4. **Resolve predictions as they come due.** This is what builds the calibration curve:
   ```bash
   uv run forge predictions --overdue
   uv run forge resolve p_XXXXX --true --note "Confirmed by announcement"
   uv run forge calibration
   ```

5. **Run evolution cycles periodically** to cull weak hypotheses and evolve the agent population:
   ```bash
   uv run forge evolve
   uv run forge leaderboard
   ```

## Development

```bash
# Run all tests
uv run pytest

# Unit tests only (fast, no I/O)
uv run pytest -m unit

# Integration tests
uv run pytest -m integration

# Coverage report
uv run pytest --cov=forge --cov-report=term-missing

# Lint
uv run ruff check forge/ tests/
```

## Architecture

See [FORGE-ARCHITECTURE.md](FORGE-ARCHITECTURE.md) for the complete technical specification.

**Pipeline:** INGEST → EXTRACT → SIMULATE → JUDGE → EVOLVE → CALIBRATE

**Two modes:**
- **Claim Testing** — fast structured analysis (steelman/redteam/judge, 3 LLM calls)
- **Scenario Simulation** — full swarm (20-100 diverse agents, multi-round interaction, consensus extraction, 90+ LLM calls)

Both feed the same hypothesis graph. Predictions get tracked for resolution. Calibration data compounds over time.

## License

MIT
