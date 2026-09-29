"""Token counting utilities using tiktoken."""

from __future__ import annotations

import functools
from typing import Any

import tiktoken


# Cache encoders per model to avoid repeated construction
@functools.lru_cache(maxsize=8)
def _get_encoder(model: str = "cl100k_base") -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding(model)


class TokenCounter:
    """Counts tokens for strings, messages, and arbitrary content."""

    def __init__(self, model: str = "cl100k_base") -> None:
        self.model = model
        self._encoder = _get_encoder(model)

    def count(self, text: str) -> int:
        """Count tokens in a plain string."""
        if not text:
            return 0
        return len(self._encoder.encode(text))

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        """
        Count tokens in an OpenAI-style message list.
        Each message has 'role' and 'content' keys.
        """
        total = 0
        for msg in messages:
            # 4 tokens per message overhead (role + formatting)
            total += 4
            total += self.count(msg.get("content", ""))
            if "name" in msg:
                total += self.count(msg["name"])
        return total

    def count_messages_detail(
        self, messages: list[dict[str, str]]
    ) -> dict[str, Any]:
        """Return a detailed per-message token breakdown."""
        details: list[dict[str, Any]] = []
        total = 0
        for i, msg in enumerate(messages):
            overhead = 4
            content_tokens = self.count(msg.get("content", ""))
            name_tokens = self.count(msg["name"]) if "name" in msg else 0
            msg_total = overhead + content_tokens + name_tokens
            total += msg_total
            details.append({
                "index": i,
                "role": msg.get("role", "unknown"),
                "overhead": overhead,
                "content_tokens": content_tokens,
                "name_tokens": name_tokens,
                "total": msg_total,
                "preview": (msg.get("content", "")[:80] + "…")
                if len(msg.get("content", "")) > 80
                else msg.get("content", ""),
            })
        return {"total": total, "messages": details}
