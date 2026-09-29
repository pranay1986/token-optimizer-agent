"""Budget manager — monitors and enforces token budgets with auto-summarization."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from .token_counter import TokenCounter


class BudgetAction(Enum):
    """Actions the budget manager can take when approaching limits."""

    NONE = "none"
    WARN = "warn"
    SUMMARIZE = "summarize"
    TRUNCATE = "truncate"
    HARD_STOP = "hard_stop"


@dataclass
class BudgetConfig:
    """Configuration for budget enforcement."""

    # Soft limit — trigger summarization
    soft_limit: int = 8000
    # Hard limit — trigger truncation or stop
    hard_limit: int = 12000
    # Fraction of soft_limit at which to warn (0.0–1.0)
    warn_threshold: float = 0.75
    # Fraction of hard_limit at which to hard-stop (0.0–1.0)
    hard_stop_threshold: float = 0.95
    # Minimum messages to keep when summarizing
    min_messages_to_keep: int = 3
    # Fraction of messages to keep when summarizing (0.0–1.0)
    keep_ratio: float = 0.5


@dataclass
class BudgetStatus:
    """Current budget status snapshot."""

    total_tokens: int
    soft_limit: int
    hard_limit: int
    usage_fraction: float
    action: BudgetAction
    message: str


@dataclass
class BudgetEvent:
    """Record of a budget-related event."""

    action: BudgetAction
    tokens_before: int
    tokens_after: int
    messages_before: int
    messages_after: int
    detail: str = ""


class BudgetManager:
    """
    Monitors token usage against configured budgets and triggers
    summarization or truncation when limits are approached.
    """

    def __init__(
        self,
        config: BudgetConfig | None = None,
        counter: TokenCounter | None = None,
        summarizer: Callable[[list[dict[str, str]]], str] | None = None,
    ) -> None:
        self.config = config or BudgetConfig()
        self.counter = counter or TokenCounter()
        self.summarizer = summarizer or self._default_summarizer
        self.events: list[BudgetEvent] = []
        self._total_processed = 0

    def check(self, messages: list[dict[str, str]]) -> BudgetStatus:
        """Check current token usage and determine the appropriate action."""
        total = self.counter.count_messages(messages)
        self._total_processed += total

        soft = self.config.soft_limit
        hard = self.config.hard_limit
        fraction = total / hard if hard > 0 else 0.0

        if fraction >= self.config.hard_stop_threshold:
            action = BudgetAction.HARD_STOP
            msg = f"Hard stop: {total}/{hard} tokens ({fraction:.0%})"
        elif fraction >= 1.0:
            action = BudgetAction.TRUNCATE
            msg = f"Over budget: {total}/{hard} tokens — truncating"
        elif total >= soft:
            action = BudgetAction.SUMMARIZE
            msg = f"Soft limit reached: {total}/{soft} tokens — summarizing"
        elif total >= soft * self.config.warn_threshold:
            action = BudgetAction.WARN
            msg = f"Approaching limit: {total}/{soft} tokens"
        else:
            action = BudgetAction.NONE
            msg = f"Within budget: {total}/{hard} tokens"

        return BudgetStatus(
            total_tokens=total,
            soft_limit=soft,
            hard_limit=hard,
            usage_fraction=fraction,
            action=action,
            message=msg,
        )

    def enforce(
        self, messages: list[dict[str, str]]
    ) -> tuple[list[dict[str, str]], BudgetStatus]:
        """
        Check budget and apply the appropriate action.
        Returns the (possibly modified) messages and the status.
        """
        status = self.check(messages)

        if status.action == BudgetAction.NONE:
            return messages, status

        if status.action == BudgetAction.WARN:
            return messages, status

        if status.action == BudgetAction.SUMMARIZE:
            new_messages = self._summarize_messages(messages)
            new_status = self.check(new_messages)
            self.events.append(BudgetEvent(
                action=BudgetAction.SUMMARIZE,
                tokens_before=status.total_tokens,
                tokens_after=new_status.total_tokens,
                messages_before=len(messages),
                messages_after=len(new_messages),
                detail=f"Reduced from {status.total_tokens} to {new_status.total_tokens} tokens",
            ))
            return new_messages, new_status

        if status.action == BudgetAction.TRUNCATE:
            new_messages = self._truncate_messages(messages)
            new_status = self.check(new_messages)
            self.events.append(BudgetEvent(
                action=BudgetAction.TRUNCATE,
                tokens_before=status.total_tokens,
                tokens_after=new_status.total_tokens,
                messages_before=len(messages),
                messages_after=len(new_messages),
                detail=f"Truncated from {status.total_tokens} to {new_status.total_tokens} tokens",
            ))
            return new_messages, new_status

        # HARD_STOP — return empty with a system message
        stop_msg = [{
            "role": "system",
            "content": (
                f"[Budget hard-stop triggered: {status.total_tokens}/{status.hard_limit} tokens. "
                "Conversation reset. Please restart with a shorter context.]"
            ),
        }]
        self.events.append(BudgetEvent(
            action=BudgetAction.HARD_STOP,
            tokens_before=status.total_tokens,
            tokens_after=self.counter.count_messages(stop_msg),
            messages_before=len(messages),
            messages_after=1,
            detail="Hard stop — conversation reset",
        ))
        return stop_msg, self.check(stop_msg)

    def _summarize_messages(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Summarize older messages, keeping recent ones intact."""
        if len(messages) <= self.config.min_messages_to_keep:
            return messages

        keep_count = max(
            self.config.min_messages_to_keep,
            int(len(messages) * self.config.keep_ratio),
        )

        # Always keep system messages
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        if len(non_system) <= keep_count:
            return messages

        to_summarize = non_system[:-keep_count]
        to_keep = non_system[-keep_count:]

        summary_text = self.summarizer(to_summarize)
        summary_msg = {
            "role": "system",
            "content": f"[Summary of earlier conversation]\n{summary_text}",
        }

        return system_msgs + [summary_msg] + to_keep

    def _truncate_messages(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Truncate messages from the beginning to fit within budget."""
        if not messages:
            return messages

        # Always keep system messages
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        # Keep removing oldest non-system messages until under budget
        while non_system:
            candidate = system_msgs + non_system
            total = self.counter.count_messages(candidate)
            if total <= self.config.soft_limit:
                return candidate
            non_system = non_system[1:]

        return system_msgs

    @staticmethod
    def _default_summarizer(messages: list[dict[str, str]]) -> str:
        """
        A simple extractive summarizer.
        In production, replace with an LLM-based summarizer.
        """
        if not messages:
            return "(no earlier conversation)"

        # Extract first sentence or first 200 chars from each message
        parts: list[str] = []
        for msg in messages:
            content = msg.get("content", "").strip()
            if not content:
                continue
            # Take first sentence or first 200 chars
            first_sentence = content.split(". ")[0]
            if len(first_sentence) > 200:
                first_sentence = first_sentence[:200] + "…"
            parts.append(f"- [{msg.get('role', 'unknown')}] {first_sentence}")

        return "\n".join(parts) if parts else "(no content to summarize)"

    def get_stats(self) -> dict:
        """Return budget management statistics."""
        return {
            "total_tokens_processed": self._total_processed,
            "events_count": len(self.events),
            "events": [
                {
                    "action": e.action.value,
                    "tokens_before": e.tokens_before,
                    "tokens_after": e.tokens_after,
                    "reduction": e.tokens_before - e.tokens_after,
                    "detail": e.detail,
                }
                for e in self.events
            ],
        }
