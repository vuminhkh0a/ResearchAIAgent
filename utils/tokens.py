"""Accurate token estimation using Qwen tokenizer for context-window checks."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _get_tokenizer():
    """Load Qwen3 tokenizer once (cached)."""
    try:
        from transformers import AutoTokenizer
        # Use Qwen3 tokenizer - works for all Qwen3 variants
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct", trust_remote_code=True)
    except Exception:
        # Fallback to rough estimation
        return None


def estimate_tokens(text: str) -> int:
    """Accurate token count using Qwen3 tokenizer."""
    if not text:
        return 0
    
    tokenizer = _get_tokenizer()
    if tokenizer is None:
        return max(1, len(text) // 4)  # Fallback
    
    return len(tokenizer.encode(text, add_special_tokens=False))


def trim_to_tokens(text: str, max_tokens: int) -> str:
    """Trim text to fit within max_tokens using accurate tokenizer."""
    if estimate_tokens(text) <= max_tokens:
        return text
    
    tokenizer = _get_tokenizer()
    if tokenizer is None:
        max_chars = max(0, max_tokens * 4)
        return text[:max_chars] + "\n\n[truncated to fit the context window]"
    
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    
    trimmed = tokenizer.decode(tokens[:max_tokens])
    return trimmed + "\n\n[truncated to fit the context window]"