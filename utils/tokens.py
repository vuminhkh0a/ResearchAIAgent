"""Lightweight token estimation for context-window checks."""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Approximate token count without a model-specific tokenizer.

    Qwen tokenizers vary; ~4 characters per token is accurate enough for V1
    to decide whether Map-Reduce or truncation is needed.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def trim_to_tokens(text: str, max_tokens: int) -> str:
    if estimate_tokens(text) <= max_tokens:
        return text
    max_chars = max(0, max_tokens * 4)
    return text[:max_chars] + "\n\n[truncated to fit the context window]"
