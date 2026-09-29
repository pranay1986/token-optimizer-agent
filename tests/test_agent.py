"""Tests for the token optimizer agent."""

from token_optimizer import (
    AgentConfig,
    BudgetConfig,
    BudgetManager,
    ContextOptimizer,
    SemanticCache,
    TokenCounter,
    TokenOptimizerAgent,
)


class TestTokenCounter:
    def test_count_basic(self):
        counter = TokenCounter()
        assert counter.count("hello world") > 0

    def test_count_empty(self):
        counter = TokenCounter()
        assert counter.count("") == 0

    def test_count_messages(self):
        counter = TokenCounter()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        assert counter.count_messages(messages) > 0

    def test_count_messages_detail(self):
        counter = TokenCounter()
        messages = [{"role": "user", "content": "hello"}]
        detail = counter.count_messages_detail(messages)
        assert detail["total"] > 0
        assert len(detail["messages"]) == 1


class TestBudgetManager:
    def test_within_budget(self):
        bm = BudgetManager(config=BudgetConfig(soft_limit=1000, hard_limit=2000))
        messages = [{"role": "user", "content": "hello"}]
        status = bm.check(messages)
        assert status.action.value == "none"

    def test_warn_threshold(self):
        bm = BudgetManager(config=BudgetConfig(soft_limit=100, hard_limit=200))
        messages = [{"role": "user", "content": "word " * 50}]
        status = bm.check(messages)
        assert status.action.value in ("warn", "summarize", "truncate", "hard_stop")

    def test_enforce_returns_messages(self):
        bm = BudgetManager(config=BudgetConfig(soft_limit=1000, hard_limit=2000))
        messages = [{"role": "user", "content": "hello"}]
        result, status = bm.enforce(messages)
        assert isinstance(result, list)
        assert status.total_tokens > 0


class TestContextOptimizer:
    def test_deduplication(self):
        optimizer = ContextOptimizer()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "hello"},
        ]
        result, opt_result = optimizer.optimize(messages, strategies=["dedup"])
        assert len(result) == 1
        assert "dedup" in opt_result.strategies_applied

    def test_whitespace_normalization(self):
        optimizer = ContextOptimizer()
        messages = [
            {"role": "user", "content": "hello\n\n\n\nworld"},
        ]
        result, opt_result = optimizer.optimize(messages, strategies=["whitespace"])
        assert "\n\n\n" not in result[0]["content"]

    def test_merge_system_messages(self):
        optimizer = ContextOptimizer()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "system", "content": "You are concise."},
        ]
        result, opt_result = optimizer.optimize(messages, strategies=["merge_sys"])
        assert len(result) == 1
        assert "helpful" in result[0]["content"]
        assert "concise" in result[0]["content"]


class TestSemanticCache:
    def test_cache_miss(self):
        cache = SemanticCache()
        result, sim = cache.get("unique prompt")
        assert result is None

    def test_cache_hit(self):
        cache = SemanticCache(similarity_threshold=0.8)
        cache.put("hello world", "response")
        result, sim = cache.get("hello world")
        assert result == "response"
        assert sim >= 0.8

    def test_cache_stats(self):
        cache = SemanticCache()
        cache.put("test", "response")
        stats = cache.get_stats()
        assert stats["entries"] == 1


class TestTokenOptimizerAgent:
    def test_process_basic(self):
        agent = TokenOptimizerAgent()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
        ]
        result = agent.process(messages)
        assert result.total_tokens_before > 0
        assert result.total_tokens_after > 0

    def test_process_with_optimization(self):
        agent = TokenOptimizerAgent(config=AgentConfig(auto_optimize=True))
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
        ]
        result = agent.process(messages)
        assert result.optimization_result is not None

    def test_cache_hit_on_repeat(self):
        agent = TokenOptimizerAgent()
        messages = [{"role": "user", "content": "What is Python?"}]
        # First call — cache miss
        result1 = agent.process(messages)
        assert not result1.cache_hit
        # Manually cache the response
        agent.cache_response("What is Python?", "Python is a programming language.")
        # Second call — should hit cache
        result2 = agent.process(messages)
        assert result2.cache_hit

    def test_get_stats(self):
        agent = TokenOptimizerAgent()
        stats = agent.get_stats()
        assert "cache" in stats
        assert "budget" in stats
        assert "config" in stats
