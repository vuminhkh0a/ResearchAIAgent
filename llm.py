"""Shared LLM factory - supports vLLM (optimized) and Ollama (legacy)."""

from __future__ import annotations

from config.settings import settings


def get_llm(*, temperature: float | None = None):
    """Get LLM instance based on configured backend with auto-fallback."""
    if settings.llm_backend == "vllm":
        try:
            from llm_vllm import get_vllm_model
            return get_vllm_model()
        except Exception as e:
            print(f"vLLM failed ({e}), falling back to Ollama...")
            settings.llm_backend = "ollama"
    
    from utils.ollama import require_ollama
    from langchain_ollama import ChatOllama
    
    require_ollama()
    kwargs = {
        "model": settings.ollama_model,
        "base_url": settings.ollama_base_url,
        "temperature": settings.llm_temperature if temperature is None else temperature,
    }
    try:
        return ChatOllama(**kwargs, reasoning=False)
    except TypeError:
        return ChatOllama(**kwargs)