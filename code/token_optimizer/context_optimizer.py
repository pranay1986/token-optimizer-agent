"""Context optimizer — deduplication, compression, and pruning strategies."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from .token_counter import TokenCounter


@dataclass
class OptimizationResult:
    """Result of a context optimization pass."""

    original_tokens: int
    optimized_tokens: int
    strategies_applied: list[str] = field(default_factory=list)
    details: list[str] = field(default_factory=list)

    @property
    def reduction(self) -> int:
        return self.original_tokens - self.optimized_tokens

    @property
    def reduction_pct(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return (self.reduction / self.original_tokens) * 100


class ContextOptimizer:
    """
    Applies various strategies to reduce token count in message lists:
      - Deduplication of repeated content
      - Whitespace normalization
      - Redundant system message merging
      - Tool result truncation
      - Code block pruning
    """

    def __init__(self, counter: TokenCounter | None = None) -> None:
        self.counter = counter or TokenCounter()

    def optimize(
        self,
        messages: list[dict[str, str]],
        strategies: list[str] | None = None,
    ) -> tuple[list[dict[str, str]], OptimizationResult]:
        """
        Apply optimization strategies to a message list.

        Strategies (apply in order):
          - "dedup"       — remove duplicate messages
          - "whitespace"  — normalize excessive whitespace
          - "merge_sys"   — merge consecutive system messages
          - "trunc_tools" — truncate long tool results
          - "prune_code"  — remove or shrink large code blocks
        """
        all_strategies = ["dedup", "whitespace", "merge_sys", "trunc_tools", "prune_code"]
        active = strategies or all_strategies

        original_tokens = self.counter.count_messages(messages)
        current = [dict(m) for m in messages]  # deep copy
        applied: list[str] = []
        details: list[str] = []

        for strategy in active:
            before = self.counter.count_messages(current)

            if strategy == "dedup":
                current = self._deduplicate(current)
            elif strategy == "whitespace":
                current = self._normalize_whitespace(current)
            elif strategy == "merge_sys":
                current = self._merge_system_messages(current)
            elif strategy == "trunc_tools":
                current = self._truncate_tool_results(current)
            elif strategy == "prune_code":
                current = self._prune_code_blocks(current)
            else:
                continue

            after = self.counter.count_messages(current)
            if after < before:
                applied.append(strategy)
                details.append(
                    f"{strategy}: {before} → {after} tokens "
                    f"(−{before - after}, −{(before - after) / before * 100:.1f}%)"
                )

        final_tokens = self.counter.count_messages(current)
        result = OptimizationResult(
            original_tokens=original_tokens,
            optimized_tokens=final_tokens,
            strategies_applied=applied,
            details=details,
        )
        return current, result

    # ── Strategy implementations ────────────────────────────────────

    def _deduplicate(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Remove consecutive duplicate messages."""
        if not messages:
            return messages

        result = [messages[0]]
        for msg in messages[1:]:
            if (
                msg.get("role") == result[-1].get("role")
                and msg.get("content") == result[-1].get("content")
            ):
                continue
            result.append(msg)
        return result

    def _normalize_whitespace(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Collapse multiple blank lines and trailing whitespace."""
        for msg in messages:
            content = msg.get("content", "")
            # Collapse 3+ newlines to 2
            content = re.sub(r"\n{3,}", "\n\n", content)
            # Strip trailing whitespace per line
            content = "\n".join(line.rstrip() for line in content.split("\n"))
            msg["content"] = content
        return messages

    def _merge_system_messages(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Merge consecutive system messages into one."""
        if not messages:
            return messages

        result: list[dict[str, str]] = []
        for msg in messages:
            if (
                msg.get("role") == "system"
                and result
                and result[-1].get("role") == "system"
            ):
                result[-1]["content"] = (
                    result[-1].get("content", "") + "\n" + msg.get("content", "")
                )
            else:
                result.append(dict(msg))
        return result

    def _truncate_tool_results(
        self,
        messages: list[dict[str, str]],
        max_chars: int = 2000,
    ) -> list[dict[str, str]]:
        """Truncate tool result messages that exceed max_chars."""
        for msg in messages:
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if len(content) > max_chars:
                    half = max_chars // 2
                    msg["content"] = (
                        content[:half]
                        + f"\n… [truncated {len(content) - max_chars} chars] …\n"
                        + content[-half:]
                    )
        return messages

    def _prune_code_blocks(
        self,
        messages: list[dict[str, str]],
        max_code_lines: int = 50,
    ) -> list[dict[str, str]]:
        """Truncate large code blocks within messages."""
        for msg in messages:
            content = msg.get("content", "")
            if "```" not in content:
                continue

            # Find code blocks and truncate if too long
            parts = content.split("```")
            # parts[0] is before first code block, parts[1] is code, etc.
            for i in range(1, len(parts), 2):
                code = parts[i]
                lines = code.split("\n")
                if len(lines) > max_code_lines:
                    kept = max_code_lines // 2
                    truncated = (
                        "\n".join(lines[:kept])
                        + f"\n// … [{len(lines) - max_code_lines} lines omitted] …\n"
                        + "\n".join(lines[-kept:])
                    )
                    parts[i] = truncated
            msg["content"] = "```".join(parts)
        return messages
