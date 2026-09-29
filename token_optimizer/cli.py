"""CLI interface for the token optimizer agent."""

from __future__ import annotations

import argparse
import json
import sys

from .agent import AgentConfig, TokenOptimizerAgent
from .token_counter import TokenCounter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Token Optimizer Agent — reduce LLM token usage"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── count command ──────────────────────────────────────────────
    count_p = sub.add_parser("count", help="Count tokens in a message list (JSON file)")
    count_p.add_argument("file", help="Path to JSON file containing messages array")
    count_p.add_argument("--model", default="cl100k_base", help="Token model")

    # ── optimize command ───────────────────────────────────────────
    opt_p = sub.add_parser("optimize", help="Optimize a message list")
    opt_p.add_argument("file", help="Path to JSON file containing messages array")
    opt_p.add_argument("--soft-limit", type=int, default=8000)
    opt_p.add_argument("--hard-limit", type=int, default=12000)
    opt_p.add_argument("--no-cache", action="store_true")
    opt_p.add_argument("--no-optimize", action="store_true")
    opt_p.add_argument("--no-budget", action="store_true")
    opt_p.add_argument("--output", "-o", help="Output file (default: stdout)")

    # ── stats command ──────────────────────────────────────────────
    stats_p = sub.add_parser("stats", help="Show agent statistics")

    # ── demo command ───────────────────────────────────────────────
    demo_p = sub.add_parser("demo", help="Run a demonstration")

    args = parser.parse_args()

    if args.command == "count":
        _cmd_count(args)
    elif args.command == "optimize":
        _cmd_optimize(args)
    elif args.command == "stats":
        _cmd_stats(args)
    elif args.command == "demo":
        _cmd_demo(args)


def _cmd_count(args: argparse.Namespace) -> None:
    with open(args.file) as f:
        messages = json.load(f)
    counter = TokenCounter(model=args.model)
    detail = counter.count_messages_detail(messages)
    print(json.dumps(detail, indent=2))


def _cmd_optimize(args: argparse.Namespace) -> None:
    with open(args.file) as f:
        messages = json.load(f)

    config = AgentConfig(
        soft_limit=args.soft_limit,
        hard_limit=args.hard_limit,
    )
    agent = TokenOptimizerAgent(config=config)
    result = agent.process(
        messages,
        use_cache=not args.no_cache,
        use_optimizer=not args.no_optimize,
        use_budget=not args.no_budget,
    )

    output = {
        "tokens_before": result.total_tokens_before,
        "tokens_after": result.total_tokens_after,
        "tokens_saved": result.tokens_saved,
        "savings_pct": round(result.savings_pct, 2),
        "cache_hit": result.cache_hit,
        "cache_similarity": round(result.cache_similarity, 4),
        "budget_action": result.budget_status.action.value,
        "budget_message": result.budget_status.message,
        "optimization": {
            "strategies_applied": result.optimization_result.strategies_applied
            if result.optimization_result
            else [],
            "details": result.optimization_result.details
            if result.optimization_result
            else [],
        },
        "messages": result.messages,
    }

    text = json.dumps(output, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"Written to {args.output}")
    else:
        print(text)


def _cmd_stats(args: argparse.Namespace) -> None:
    agent = TokenOptimizerAgent()
    print(json.dumps(agent.get_stats(), indent=2))


def _cmd_demo(args: argparse.Namespace) -> None:
    """Run a demonstration of all three optimization strategies."""
    print("=" * 70)
    print("  Token Optimizer Agent — Demo")
    print("=" * 70)

    # Create a sample conversation with redundancy
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "system", "content": "You are a helpful assistant."},  # duplicate
        {"role": "user", "content": "What is Python?"},
        {
            "role": "assistant",
            "content": (
                "Python is a high-level programming language.\n\n\n\n"
                "It is widely used for web development, data science, and AI."
            ),
        },
        {"role": "user", "content": "What is Python?"},  # duplicate question
        {
            "role": "assistant",
            "content": (
                "Python is a high-level programming language.\n\n\n\n"
                "It is widely used for web development, data science, and AI."
            ),
        },
        {
            "role": "user",
            "content": "Show me a code example",
        },
        {
            "role": "assistant",
            "content": (
                "Here is an example:\n```python\n"
                + "\n".join(f"def func_{i}(): return {i}" for i in range(100))
                + "\n```"
            ),
        },
    ]

    counter = TokenCounter()
    before = counter.count_messages(messages)
    print(f"\nOriginal messages: {len(messages)}")
    print(f"Original tokens:   {before}")

    # Process through agent
    agent = TokenOptimizerAgent()
    result = agent.process(messages)

    print(f"\n--- Results ---")
    print(f"Optimized tokens:  {result.total_tokens_after}")
    print(f"Tokens saved:      {result.tokens_saved} ({result.savings_pct:.1f}%)")
    print(f"Cache hit:         {result.cache_hit}")
    print(f"Budget action:     {result.budget_status.action.value}")

    if result.optimization_result:
        print(f"\nStrategies applied: {result.optimization_result.strategies_applied}")
        for detail in result.optimization_result.details:
            print(f"  • {detail}")

    # Demonstrate caching
    print(f"\n--- Cache Demo ---")
    # Process the same messages again — should hit cache
    result2 = agent.process(messages)
    print(f"Second run cache hit: {result2.cache_hit}")
    print(f"Cache similarity:     {result2.cache_similarity:.4f}")

    # Show stats
    print(f"\n--- Agent Stats ---")
    stats = agent.get_stats()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
