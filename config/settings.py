"""Application settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    # LLM Backend: "ollama" or "vllm"
    llm_backend: str = os.getenv("LLM_BACKEND", "vllm").lower()
    
    # Ollama settings (legacy)
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    
    # vLLM settings
    vllm_model: str = os.getenv("VLLM_MODEL", "Qwen/Qwen3-4B-Instruct")
    vllm_tensor_parallel_size: int = _int("VLLM_TENSOR_PARALLEL_SIZE", 1)
    vllm_gpu_memory_utilization: float = _float("VLLM_GPU_MEMORY_UTILIZATION", 0.85)
    vllm_max_model_len: int = _int("VLLM_MAX_MODEL_LEN", 8192)
    vllm_dtype: str = os.getenv("VLLM_DTYPE", "half")
    vllm_quantization: str = os.getenv("VLLM_QUANTIZATION", "awq")
    vllm_enforce_eager: bool = _bool("VLLM_ENFORCE_EAGER", False)
    vllm_enable_prefix_caching: bool = _bool("VLLM_ENABLE_PREFIX_CACHING", True)
    vllm_max_num_batched_tokens: int = _int("VLLM_MAX_NUM_BATCHED_TOKENS", 4096)
    
    # Embeddings
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_use_onnx: bool = _bool("EMBEDDING_USE_ONNX", True)
    embedding_batch_size: int = _int("EMBEDDING_BATCH_SIZE", 32)

    # Text processing
    chunk_size: int = _int("CHUNK_SIZE", 1500)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 200)
    retrieval_k: int = _int("RETRIEVAL_K", 4)
    max_context_tokens: int = _int("MAX_CONTEXT_TOKENS", 4000)
    context_window_tokens: int = _int("CONTEXT_WINDOW_TOKENS", 8192)
    reserved_response_tokens: int = _int("RESERVED_RESPONSE_TOKENS", 2048)

    # Search
    search_provider: str = os.getenv("SEARCH_PROVIDER", "duckduckgo").lower()
    search_max_results: int = _int("SEARCH_MAX_RESULTS", 5)
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")

    # Generation
    llm_temperature: float = _float("LLM_TEMPERATURE", 0.1)

    # Paths
    chroma_dir: Path = ROOT_DIR / "data" / "chroma"
    upload_dir: Path = ROOT_DIR / "data" / "uploads"
    ingested_index_path: Path = ROOT_DIR / "data" / "ingested.json"


settings = Settings()
settings.chroma_dir.mkdir(parents=True, exist_ok=True)
settings.upload_dir.mkdir(parents=True, exist_ok=True)
