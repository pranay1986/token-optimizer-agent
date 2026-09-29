# Token Optimizer Agent

An agentic Python system that reduces LLM token usage through three complementary strategies:

| Strategy | What it does | When it helps |
|---|---|---|
| **Semantic Cache** | Detects similar prompts and returns cached responses instead of calling the LLM | Repeated or near-duplicate queries |
| **Context Optimizer** | Deduplicates messages, normalizes whitespace, merges system messages, truncates tool results, prunes large code blocks | Bloated conversation history |
| **Budget Manager** | Monitors token count and auto-summarizes or truncates when approaching limits | Long conversations at risk of exceeding context windows |

## Installation

```bash
cd token-optimizer
pip install -e .
```

## Quick Start

### CLI

```bash
# Count tokens in a message file
python -m token_optimizer.cli count examples/sample_messages.json

# Optimize a message file
python -m token_optimizer.cli optimize examples/sample_messages.json

# Run the demo
python -m token_optimizer.cli demo

# Run benchmarks and generate plots
python benchmark.py
```

### Python API

```python
from token_optimizer import TokenOptimizerAgent, AgentConfig

# Configure
config = AgentConfig(
    soft_limit=8000,      # trigger summarization
    hard_limit=12000,     # trigger truncation
    cache_similarity_threshold=0.85,
    auto_optimize=True,
)

agent = TokenOptimizerAgent(config=config)

# Process messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language."},
]

result = agent.process(messages)

print(f"Tokens: {result.total_tokens_before} → {result.total_tokens_after}")
print(f"Saved:  {result.tokens_saved} ({result.savings_pct:.1f}%)")
print(f"Cache:  {'HIT' if result.cache_hit else 'miss'}")
print(f"Budget: {result.budget_status.message}")
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  TokenOptimizerAgent                 │
│                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │ SemanticCache │→│ContextOptim. │→│ BudgetMgr │ │
│  │              │  │              │  │           │ │
│  │ • Embedding  │  │ • Dedup      │  │ • Monitor │ │
│  │ • Cosine sim │  │ • Whitespace │  │ • Summarize│ │
│  │ • LRU evict  │  │ • Merge sys  │  │ • Truncate│ │
│  │              │  │ • Trunc tools│  │ • Stop    │ │
│  │              │  │ • Prune code │  │           │ │
│  └──────────────┘  └──────────────┘  └───────────┘ │
│                                                     │
│  Pipeline: cache → optimize → budget                │
└─────────────────────────────────────────────────────┘
```

## How Each Strategy Works

### 1. Semantic Cache
- Embeds prompts using bag-of-words vectors (swap in a real embedding model for production)
- Computes cosine similarity against cached entries
- If similarity ≥ threshold, returns cached response — **zero LLM tokens used**
- LRU eviction when cache is full

### 2. Context Optimizer
Five strategies applied in order:

| Strategy | Description |
|---|---|
| `dedup` | Remove consecutive duplicate messages |
| `whitespace` | Collapse 3+ newlines to 2, strip trailing whitespace |
| `merge_sys` | Merge consecutive system messages into one |
| `trunc_tools` | Truncate tool results exceeding 2000 chars (keep head + tail) |
| `prune_code` | Truncate code blocks exceeding 50 lines (keep head + tail) |

### 3. Budget Manager
Three threshold-based actions:

| Threshold | Action |
|---|---|
| ≥ 75% of soft limit | **Warn** — log only |
| ≥ soft limit | **Summarize** — compress older messages into a summary |
| ≥ hard limit | **Truncate** — drop oldest messages until under soft limit |
| ≥ 95% of hard limit | **Hard stop** — reset conversation |

## Configuration

```python
AgentConfig(
    soft_limit=8000,              # soft token budget
    hard_limit=12000,             # hard token budget
    cache_similarity_threshold=0.85,  # 0.0–1.0, higher = stricter matching
    cache_max_entries=1000,       # max cached responses
    auto_optimize=True,           # auto-apply context optimization
    optimization_strategies=None, # None = all, or list like ["dedup", "whitespace"]
)
```

## Statistics

```python
stats = agent.get_stats()
# {
#   "cache": {"entries": 42, "total_hits": 15, "total_tokens_saved": 3200, "hit_rate": 0.26},
#   "budget": {"total_tokens_processed": 58000, "events_count": 3, "events": [...]},
#   "config": {...}
# }
```

## Benchmark Results

Run `python benchmark.py` to generate plots showing:

- **Scenario comparison** — token savings across short/medium/long/code-heavy/redundant conversations
- **Model comparison** — context windows and pricing across 12 popular models
- **Cost savings** — dollar impact of 30% token reduction per model
- **Budget enforcement** — how the system behaves as conversations grow
- **Cache performance** — hit rates and tokens saved
- **Strategy breakdown** — individual contribution of each optimization strategy

## GitHub Repository

### Push to GitHub

```bash
# Create a new repository on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/token-optimizer-agent.git
git branch -M main
git push -u origin main
```

### Clone and Run

```bash
git clone https://github.com/YOUR_USERNAME/token-optimizer-agent.git
cd token-optimizer-agent
pip install -e .
python -m token_optimizer.cli demo
python benchmark.py
```

## Project Structure

```
token-optimizer/
├── token_optimizer/
│   ├── __init__.py          # Package exports
│   ├── agent.py             # Main orchestrator
│   ├── budget_manager.py    # Budget enforcement
│   ├── context_optimizer.py # Context optimization
│   ├── semantic_cache.py    # Semantic caching
│   ├── token_counter.py     # Token counting
│   └── cli.py               # CLI interface
├── tests/
│   └── test_agent.py        # Unit tests
├── examples/
│   └── sample_messages.json # Example input
├── benchmark.py             # Benchmark suite
├── benchmark_results/       # Generated plots
├── pyproject.toml           # Package config
└── README.md
```

## Production Tips

1. **Replace the embedding** in `semantic_cache.py` with a real model (OpenAI `text-embedding-3-small`, `sentence-transformers`, etc.)
2. **Replace the summarizer** in `budget_manager.py` with an LLM call for higher-quality summaries
3. **Persist the cache** to disk (Redis, SQLite) across sessions
4. **Tune thresholds** based on your model's context window and typical usage patterns
5. **Monitor stats** in production to find the sweet spot for your workload

## License

MIT
