"""Semantic cache — avoids re-processing similar prompts via embedding similarity."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .token_counter import TokenCounter


@dataclass
class CacheEntry:
    """A single cache entry."""

    key: str
    embedding: np.ndarray
    response: str
    tokens_saved: int
    timestamp: float = field(default_factory=time.time)
    hits: int = 0


class SemanticCache:
    """
    A semantic cache that stores responses keyed by embedding similarity.
    When a new prompt is similar enough to a cached one, the cached
    response is returned instead of re-processing — saving tokens.

    Uses a simple bag-of-words embedding (no external API needed).
    For production, replace with a real embedding model.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_entries: int = 1000,
        counter: TokenCounter | None = None,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.counter = counter or TokenCounter()
        self._entries: list[CacheEntry] = []
        self._total_hits = 0
        self._total_tokens_saved = 0

    def get(self, prompt: str) -> tuple[str | None, float]:
        """
        Look up a semantically similar prompt in the cache.
        Returns (cached_response, similarity) or (None, 0.0).
        """
        if not self._entries:
            return None, 0.0

        query_emb = self._embed(prompt)
        best_sim = 0.0
        best_entry: CacheEntry | None = None

        for entry in self._entries:
            sim = self._cosine_similarity(query_emb, entry.embedding)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_entry is not None and best_sim >= self.similarity_threshold:
            best_entry.hits += 1
            self._total_hits += 1
            self._total_tokens_saved += best_entry.tokens_saved
            return best_entry.response, best_sim

        return None, best_sim

    def put(self, prompt: str, response: str, tokens_saved: int | None = None) -> None:
        """Store a prompt-response pair in the cache."""
        if tokens_saved is None:
            tokens_saved = self.counter.count(prompt)

        key = self._make_key(prompt)
        embedding = self._embed(prompt)

        # Evict oldest if at capacity
        if len(self._entries) >= self.max_entries:
            self._entries.pop(0)

        self._entries.append(CacheEntry(
            key=key,
            embedding=embedding,
            response=response,
            tokens_saved=tokens_saved,
        ))

    def invalidate(self, prompt: str) -> bool:
        """Remove a cache entry by exact prompt match. Returns True if found."""
        key = self._make_key(prompt)
        for i, entry in enumerate(self._entries):
            if entry.key == key:
                self._entries.pop(i)
                return True
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        self._entries.clear()
        self._total_hits = 0
        self._total_tokens_saved = 0

    def get_stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        return {
            "entries": len(self._entries),
            "total_hits": self._total_hits,
            "total_tokens_saved": self._total_tokens_saved,
            "hit_rate": self._total_hits / (self._total_hits + len(self._entries))
            if (self._total_hits + len(self._entries)) > 0
            else 0.0,
        }

    # ── Embedding (simple bag-of-words) ─────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        """
        Create a simple bag-of-words embedding.
        Uses word frequency vectors — no external dependencies.
        For production, use a real embedding model (OpenAI, sentence-transformers, etc.)
        """
        words = self._tokenize(text)
        if not words:
            return np.zeros(256, dtype=np.float32)

        # Hash words into a fixed-size vector
        vec = np.zeros(256, dtype=np.float32)
        for word in words:
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % 256
            vec[idx] += 1.0

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple lowercase word tokenizer."""
        import re
        return re.findall(r"[a-z0-9]+", text.lower())

    @staticmethod
    def _make_key(text: str) -> str:
        """Create a deterministic cache key."""
        return hashlib.sha256(text.encode()).hexdigest()
