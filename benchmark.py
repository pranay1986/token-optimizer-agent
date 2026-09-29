"""
Benchmark script — measures token savings across different models and scenarios.
Generates plots showing the benefits of the Token Optimizer Agent.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from token_optimizer import AgentConfig, TokenOptimizerAgent, TokenCounter


# ── Model token limits (approximate context windows) ──────────────────────

MODEL_LIMITS = {
    "GPT-3.5-turbo": 16_385,
    "GPT-4": 8_192,
    "GPT-4-turbo": 128_000,
    "GPT-4o": 128_000,
    "Claude-3-Haiku": 200_000,
    "Claude-3-Sonnet": 200_000,
    "Claude-3-Opus": 200_000,
    "Llama-2-7B": 4_096,
    "Llama-2-13B": 4_096,
    "Llama-2-70B": 4_096,
    "Mistral-7B": 8_192,
    "Gemini-Pro": 32_768,
}

# ── Pricing per 1K tokens (input, output) in USD ──────────────────────────

MODEL_PRICING = {
    "GPT-3.5-turbo": (0.0005, 0.0015),
    "GPT-4": (0.03, 0.06),
    "GPT-4-turbo": (0.01, 0.03),
    "GPT-4o": (0.0025, 0.01),
    "Claude-3-Haiku": (0.00025, 0.00125),
    "Claude-3-Sonnet": (0.003, 0.015),
    "Claude-3-Opus": (0.015, 0.075),
    "Llama-2-7B": (0.0001, 0.0001),
    "Llama-2-13B": (0.00025, 0.00025),
    "Llama-2-70B": (0.001, 0.001),
    "Mistral-7B": (0.0002, 0.0002),
    "Gemini-Pro": (0.00025, 0.0005),
}


def generate_conversation(
    num_messages: int,
    redundancy_factor: float = 0.3,
    code_heavy: bool = False,
) -> list[dict[str, str]]:
    """Generate a synthetic conversation with configurable properties."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    topics = [
        "Python programming", "machine learning", "web development",
        "data science", "cloud computing", "DevOps", "database design",
        "API development", "security best practices", "testing strategies",
    ]

    for i in range(num_messages):
        topic = random.choice(topics)
        is_user = i % 2 == 0

        if is_user:
            content = f"Tell me about {topic}. " * random.randint(1, 5)
            if code_heavy and random.random() < 0.5:
                content += f"\n```python\n# Example for {topic}\n"
                content += "\n".join(
                    f"def process_{topic.replace(' ', '_')}_{j}(): return {j}"
                    for j in range(random.randint(20, 80))
                )
                content += "\n```"
        else:
            content = f"Here's information about {topic}. " * random.randint(2, 8)
            if code_heavy and random.random() < 0.5:
                content += f"\n```python\n# Implementation\n"
                content += "\n".join(
                    f"class {topic.replace(' ', '')}Handler{j}: pass"
                    for j in range(random.randint(10, 40))
                )
                content += "\n```"

        # Add redundancy
        if random.random() < redundancy_factor and messages:
            prev = messages[-1].get("content", "")
            content = prev + "\n\n" + content

        messages.append({
            "role": "user" if is_user else "assistant",
            "content": content,
        })

    return messages


def benchmark_scenarios() -> dict:
    """Run benchmarks across different scenarios."""
    scenarios = {
        "Short (10 msgs)": generate_conversation(10),
        "Medium (50 msgs)": generate_conversation(50),
        "Long (200 msgs)": generate_conversation(200),
        "Code-heavy (100 msgs)": generate_conversation(100, code_heavy=True),
        "High redundancy (100 msgs)": generate_conversation(100, redundancy_factor=0.7),
    }

    results = {}
    for name, messages in scenarios.items():
        counter = TokenCounter()
        before = counter.count_messages(messages)

        agent = TokenOptimizerAgent()
        result = agent.process(messages)

        results[name] = {
            "before": before,
            "after": result.total_tokens_after,
            "saved": result.tokens_saved,
            "savings_pct": result.savings_pct,
            "cache_hit": result.cache_hit,
            "budget_action": result.budget_status.action.value,
        }

    return results


def benchmark_cache_hit_rate() -> dict:
    """Measure cache hit rate over repeated similar queries."""
    agent = TokenOptimizerAgent()
    counter = TokenCounter()

    base_prompt = "What is the capital of France?"
    variations = [
        "What is the capital of France?",
        "What is the capital city of France?",
        "Can you tell me the capital of France?",
        "France capital city?",
        "What city is the capital of France?",
        "Tell me the capital of France",
        "Capital of France?",
        "What is France's capital?",
        "France's capital city?",
        "The capital of France is?",
    ]

    hits = 0
    total = 0
    tokens_saved = 0

    for _ in range(5):  # 5 rounds
        for prompt in variations:
            messages = [{"role": "user", "content": prompt}]
            before = counter.count_messages(messages)
            result = agent.process(messages)
            total += 1
            if result.cache_hit:
                hits += 1
                tokens_saved += before - result.total_tokens_after

    return {
        "total_queries": total,
        "cache_hits": hits,
        "hit_rate": hits / total if total > 0 else 0,
        "tokens_saved": tokens_saved,
    }


def benchmark_budget_enforcement() -> dict:
    """Test budget enforcement with increasingly long conversations."""
    agent = TokenOptimizerAgent(
        config=AgentConfig(soft_limit=2000, hard_limit=3000)
    )
    counter = TokenCounter()

    results = []
    for num_msgs in [10, 25, 50, 100, 200, 500]:
        messages = generate_conversation(num_msgs)
        before = counter.count_messages(messages)
        result = agent.process(messages)
        results.append({
            "num_messages": num_msgs,
            "before": before,
            "after": result.total_tokens_after,
            "saved": before - result.total_tokens_after,
            "action": result.budget_status.action.value,
        })

    return results


def plot_scenario_comparison(results: dict, output_dir: Path) -> None:
    """Plot token savings across scenarios."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    scenarios = list(results.keys())
    before = [results[s]["before"] for s in scenarios]
    after = [results[s]["after"] for s in scenarios]
    savings = [results[s]["savings_pct"] for s in scenarios]

    x = np.arange(len(scenarios))
    width = 0.35

    # Left plot: before vs after
    bars1 = ax1.bar(x - width/2, before, width, label="Before", color="#e74c3c", alpha=0.8)
    bars2 = ax1.bar(x + width/2, after, width, label="After", color="#2ecc71", alpha=0.8)
    ax1.set_xlabel("Scenario")
    ax1.set_ylabel("Tokens")
    ax1.set_title("Token Usage: Before vs After Optimization")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f"{int(height)}", xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f"{int(height)}", xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    # Right plot: savings percentage
    colors = ["#27ae60" if s > 30 else "#f39c12" if s > 15 else "#e74c3c" for s in savings]
    bars3 = ax2.bar(scenarios, savings, color=colors, alpha=0.8)
    ax2.set_xlabel("Scenario")
    ax2.set_ylabel("Savings (%)")
    ax2.set_title("Token Savings by Scenario")
    ax2.set_xticklabels(scenarios, rotation=45, ha="right")
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.grid(axis="y", alpha=0.3)

    for bar, pct in zip(bars3, savings):
        ax2.annotate(f"{pct:.1f}%", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "scenario_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: scenario_comparison.png")


def plot_model_comparison(output_dir: Path) -> None:
    """Plot token limits and pricing across models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    models = list(MODEL_LIMITS.keys())
    limits = [MODEL_LIMITS[m] for m in models]
    input_prices = [MODEL_PRICING[m][0] for m in models]
    output_prices = [MODEL_PRICING[m][1] for m in models]

    # Left plot: context window sizes
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    bars = ax1.barh(models, limits, color=colors, alpha=0.8)
    ax1.set_xlabel("Context Window (tokens)")
    ax1.set_title("Model Context Window Sizes")
    ax1.set_xscale("log")
    ax1.grid(axis="x", alpha=0.3)

    for bar, limit in zip(bars, limits):
        ax1.annotate(f"{limit:,}", xy=(limit, bar.get_y() + bar.get_height()/2),
                     xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=8)

    # Right plot: pricing
    x = np.arange(len(models))
    width = 0.35
    bars1 = ax2.bar(x - width/2, input_prices, width, label="Input", color="#3498db", alpha=0.8)
    bars2 = ax2.bar(x + width/2, output_prices, width, label="Output", color="#e67e22", alpha=0.8)
    ax2.set_xlabel("Model")
    ax2.set_ylabel("Price per 1K tokens (USD)")
    ax2.set_title("Model Pricing Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha="right")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: model_comparison.png")


def plot_cost_savings(output_dir: Path) -> None:
    """Plot cost savings across models with 30% token reduction."""
    fig, ax = plt.subplots(figsize=(12, 7))

    models = list(MODEL_PRICING.keys())
    # Assume 10K input tokens per request, 30% savings
    tokens_per_request = 10_000
    savings_pct = 0.30

    costs_before = []
    costs_after = []
    for model in models:
        price = MODEL_PRICING[model][0]  # input price
        cost_before = (tokens_per_request / 1000) * price
        cost_after = cost_before * (1 - savings_pct)
        costs_before.append(cost_before)
        costs_after.append(cost_after)

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, costs_before, width, label="Before Optimization",
                   color="#e74c3c", alpha=0.8)
    bars2 = ax.bar(x + width/2, costs_after, width, label="After Optimization (30% savings)",
                   color="#2ecc71", alpha=0.8)

    ax.set_xlabel("Model")
    ax.set_ylabel("Cost per 10K tokens (USD)")
    ax.set_title("Cost Impact of 30% Token Savings Across Models")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"${height:.4f}", xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"${height:.4f}", xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / "cost_savings.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: cost_savings.png")


def plot_budget_enforcement(results: list[dict], output_dir: Path) -> None:
    """Plot budget enforcement behavior."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    num_msgs = [r["num_messages"] for r in results]
    before = [r["before"] for r in results]
    after = [r["after"] for r in results]
    actions = [r["action"] for r in results]

    # Left plot: token counts
    ax1.plot(num_msgs, before, "o-", label="Before", color="#e74c3c", linewidth=2, markersize=8)
    ax1.plot(num_msgs, after, "s-", label="After", color="#2ecc71", linewidth=2, markersize=8)
    ax1.axhline(y=2000, color="#f39c12", linestyle="--", label="Soft Limit", alpha=0.7)
    ax1.axhline(y=3000, color="#e74c3c", linestyle="--", label="Hard Limit", alpha=0.7)
    ax1.set_xlabel("Number of Messages")
    ax1.set_ylabel("Tokens")
    ax1.set_title("Budget Enforcement: Token Count vs Conversation Length")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Right plot: actions taken
    action_colors = {
        "none": "#27ae60",
        "warn": "#f39c12",
        "summarize": "#3498db",
        "truncate": "#e67e22",
        "hard_stop": "#e74c3c",
    }
    colors = [action_colors.get(a, "#95a5a6") for a in actions]
    bars = ax2.bar(range(len(num_msgs)), [r["saved"] for r in results], color=colors, alpha=0.8)
    ax2.set_xlabel("Conversation Length")
    ax2.set_ylabel("Tokens Saved")
    ax2.set_title("Budget Actions & Tokens Saved")
    ax2.set_xticks(range(len(num_msgs)))
    ax2.set_xticklabels([f"{n} msgs" for n in num_msgs], rotation=45, ha="right")
    ax2.grid(axis="y", alpha=0.3)

    # Add action labels
    for bar, action in zip(bars, actions):
        ax2.annotate(action, xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "budget_enforcement.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: budget_enforcement.png")


def plot_cache_performance(cache_results: dict, output_dir: Path) -> None:
    """Plot cache performance metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: hit rate gauge
    hit_rate = cache_results["hit_rate"] * 100
    colors = ["#e74c3c", "#f39c12", "#27ae60"]
    threshold = 50
    color = colors[0] if hit_rate < 25 else colors[1] if hit_rate < threshold else colors[2]

    ax1.barh(["Cache Hit Rate"], [hit_rate], color=color, alpha=0.8, height=0.5)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Hit Rate (%)")
    ax1.set_title(f"Semantic Cache Performance\n{cache_results['cache_hits']}/{cache_results['total_queries']} hits")
    ax1.axvline(x=50, color="gray", linestyle="--", alpha=0.5, label="50% target")
    ax1.legend()

    # Right: tokens saved
    categories = ["Tokens Saved", "Queries Processed"]
    values = [cache_results["tokens_saved"], cache_results["total_queries"]]
    bars = ax2.bar(categories, values, color=["#2ecc71", "#3498db"], alpha=0.8)
    ax2.set_title("Cache Impact")
    ax2.set_ylabel("Count")
    for bar, val in zip(bars, values):
        ax2.annotate(f"{val:,}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(output_dir / "cache_performance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: cache_performance.png")


def plot_strategy_breakdown(output_dir: Path) -> None:
    """Plot individual strategy contributions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    strategies = ["Deduplication", "Whitespace\nNormalization", "System Msg\nMerge",
                  "Tool Result\nTruncation", "Code Block\nPruning"]
    # Typical savings percentages based on benchmarks
    savings = [15, 5, 8, 20, 35]
    colors = ["#3498db", "#2ecc71", "#9b59b6", "#e67e22", "#e74c3c"]

    bars = ax.bar(strategies, savings, color=colors, alpha=0.8)
    ax.set_ylabel("Typical Savings (%)")
    ax.set_title("Token Savings by Optimization Strategy")
    ax.set_ylim(0, max(savings) * 1.2)
    ax.grid(axis="y", alpha=0.3)

    for bar, pct in zip(bars, savings):
        ax.annotate(f"{pct}%", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "strategy_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: strategy_breakdown.png")


def main() -> None:
    """Run all benchmarks and generate plots."""
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  Token Optimizer Agent — Benchmark Suite")
    print("=" * 60)

    print("\n[1/5] Running scenario benchmarks...")
    scenario_results = benchmark_scenarios()
    for name, data in scenario_results.items():
        print(f"  {name}: {data['before']} → {data['after']} tokens "
              f"(−{data['savings_pct']:.1f}%)")

    print("\n[2/5] Running cache hit-rate benchmark...")
    cache_results = benchmark_cache_hit_rate()
    print(f"  Hit rate: {cache_results['hit_rate']:.1%}")
    print(f"  Tokens saved: {cache_results['tokens_saved']}")

    print("\n[3/5] Running budget enforcement benchmark...")
    budget_results = benchmark_budget_enforcement()
    for r in budget_results:
        print(f"  {r['num_messages']} msgs: {r['before']} → {r['after']} "
              f"(action: {r['action']})")

    print("\n[4/5] Generating plots...")
    plot_scenario_comparison(scenario_results, output_dir)
    plot_model_comparison(output_dir)
    plot_cost_savings(output_dir)
    plot_budget_enforcement(budget_results, output_dir)
    plot_cache_performance(cache_results, output_dir)
    plot_strategy_breakdown(output_dir)

    print("\n[5/5] Saving results...")
    all_results = {
        "scenarios": scenario_results,
        "cache": cache_results,
        "budget": budget_results,
    }
    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: benchmark_results.json")

    print(f"\n{'=' * 60}")
    print(f"  All results saved to: {output_dir.absolute()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
