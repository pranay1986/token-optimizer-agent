"""Main agent — orchestrates budget management, context optimization, and semantic caching."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .budget_manager import BudgetConfig, BudgetManager, BudgetStatus
from .context_optimizer import ContextOptimizer, OptimizationResult
from .semantic_cache import SemanticCache
from .token_counter import TokenCounter


@dataclass
class AgentConfig:
    """Configuration for the token optimizer agent."""

    # Budget settings
    soft_limit: int = 8000
    hard_limit: int = 12000
    # Cache settings
    cache_similarity_threshold: float = 0.85
    cache_max_entries: int = 1000
    # Optimization settings
    auto_optimize: bool = True
    optimization_strategies: list[str] | None = None


@dataclass
class ProcessingResult:
    """Result of processing a message list through the agent."""

    messages: list[dict[str, str]]
    budget_status: BudgetStatus
    optimization_result: OptimizationResult | None
    cache_hit: bool
    cache_similarity: float
    total_tokens_before: int
    total_tokens_after: int

    @property
    def tokens_saved(self) -> int:
        return self.total_tokens_before - self.total_tokens_after

    @property
    def savings_pct(self) -> float:
        if self.total_tokens_before == 0:
            return 0.0
        return (self.tokens_saved / self.total_tokens_before) * 100


class TokenOptimizerAgent:
    """
    Agentic token optimizer that combines three strategies:

    1. **Semantic Cache** — checks if a similar prompt was already processed
       and returns the cached response, skipping LLM calls entirely.
    2. **Context Optimizer** — applies deduplication, whitespace normalization,
       system message merging, tool result truncation, and code block pruning.
    3. **Budget Manager** — monitors token usage and triggers summarization
       or truncation when approaching configured limits.

    Usage:
        agent = TokenOptimizerAgent()
        result = agent.process(messages)
        print(f"Saved {result.tokens_saved} tokens ({result.savings_pct:.1f}%)")
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        llm_callback: Callable[[list[dict[str, str]]], str] | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.llm_callback = llm_callback

        self.counter = TokenCounter()
        self.cache = SemanticCache(
            similarity_threshold=self.config.cache_similarity_threshold,
            max_entries=self.config.cache_max_entries,
            counter=self.counter,
        )
        self.optimizer = ContextOptimizer(counter=self.counter)
        self.budget_manager = BudgetManager(
            config=BudgetConfig(
                soft_limit=self.config.soft_limit,
                hard_limit=self.config.hard_limit,
            ),
            counter=self.counter,
        )

    def process(
        self,
        messages: list[dict[str, str]],
        use_cache: bool = True,
        use_optimizer: bool | None = None,
        use_budget: bool = True,
    ) -> ProcessingResult:
        """
        Process a message list through the full optimization pipeline.

        Pipeline order:
          1. Semantic cache check (on the last user message)
          2. Context optimization
          3. Budget enforcement
        """
        if use_optimizer is None:
            use_optimizer = self.config.auto_optimize

        tokens_before = self.counter.count_messages(messages)

        # ── Step 1: Semantic cache ──────────────────────────────────
        cache_hit = False
        cache_similarity = 0.0

        if use_cache:
            # Use the last user message as the cache query
            last_user_msg = self._get_last_user_message(messages)
            if last_user_msg:
                cached_response, cache_similarity = self.cache.get(last_user_msg)
                if cached_response is not None:
                    cache_hit = True
                    # Replace the last user message with the cached response
                    messages = self._apply_cached_response(messages, cached_response)
                    tokens_after = self.counter.count_messages(messages)
                    return ProcessingResult(
                        messages=messages,
                        budget_status=self.budget_manager.check(messages),
                        optimization_result=None,
                        cache_hit=True,
                        cache_similarity=cache_similarity,
                        total_tokens_before=tokens_before,
                        total_tokens_after=tokens_after,
                    )

        # ── Step 2: Context optimization ────────────────────────────
        opt_result: OptimizationResult | None = None
        if use_optimizer:
            messages, opt_result = self.optimizer.optimize(
                messages, strategies=self.config.optimization_strategies
            )

        # ── Step 3: Budget enforcement ──────────────────────────────
        budget_status: BudgetStatus | None = None
        if use_budget:
            messages, budget_status = self.budget_manager.enforce(messages)
        else:
            budget_status = self.budget_manager.check(messages)

        tokens_after = self.counter.count_messages(messages)

        return ProcessingResult(
            messages=messages,
            budget_status=budget_status,
            optimization_result=opt_result,
            cache_hit=False,
            cache_similarity=cache_similarity,
            total_tokens_before=tokens_before,
            total_tokens_after=tokens_after,
        )

    def cache_response(
        self, prompt: str, response: str, tokens_saved: int | None = None
    ) -> None:
        """Manually add a prompt-response pair to the semantic cache."""
        self.cache.put(prompt, response, tokens_saved)

    def get_stats(self) -> dict[str, Any]:
        """Return combined statistics from all components."""
        return {
            "cache": self.cache.get_stats(),
            "budget": self.budget_manager.get_stats(),
            "config": {
                "soft_limit": self.config.soft_limit,
                "hard_limit": self.config.hard_limit,
                "cache_similarity_threshold": self.config.cache_similarity_threshold,
                "auto_optimize": self.config.auto_optimize,
            },
        }

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _get_last_user_message(messages: list[dict[str, str]]) -> str | None:
        """Extract the content of the last user message."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return None

    @staticmethod
    def _apply_cached_response(
        messages: list[dict[str, str]], cached_response: str
    ) -> list[dict[str, str]]:
        """Replace the last user message with the cached response."""
        result = list(messages)
        for i in range(len(result) - 1, -1, -1):
            if result[i].get("role") == "user":
                result[i] = {"role": "assistant", "content": cached_response}
                break
        return result
