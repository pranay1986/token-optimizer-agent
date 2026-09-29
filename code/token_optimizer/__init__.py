"""
Token Optimizer Agent
=====================
An agentic system that reduces LLM token usage through:
  1. Budget management  — monitors and enforces token budgets
  2. Context optimization — deduplication, compression, pruning
  3. Semantic caching  — avoids re-processing similar prompts
"""

from .agent import AgentConfig, TokenOptimizerAgent
from .budget_manager import BudgetConfig, BudgetManager
from .context_optimizer import ContextOptimizer
from .semantic_cache import SemanticCache
from .token_counter import TokenCounter

__all__ = [
    "AgentConfig",
    "BudgetConfig",
    "TokenOptimizerAgent",
    "BudgetManager",
    "ContextOptimizer",
    "SemanticCache",
    "TokenCounter",
]

__version__ = "1.0.0"
