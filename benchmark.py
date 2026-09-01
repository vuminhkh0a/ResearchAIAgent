#!/usr/bin/env python
"""Performance benchmark script for the research assistant."""

from __future__ import annotations

import time
from pathlib import Path

from agent.graph import run_agent, run_agent_stream
from agent.state import AgentState
from langchain_core.messages import HumanMessage
from config.settings import settings
from summarization.map_reduce import summarize_text
from rag.embeddings import get_embeddings


def benchmark_llm_inference():
    """Benchmark raw LLM inference speed."""
    from llm import get_llm
    
    llm = get_llm()
    prompt = "Explain quantum computing in simple terms."
    messages = [HumanMessage(content=prompt)]
    
    # Warmup
    _ = llm.invoke(messages)
    
    # Benchmark
    times = []
    for _ in range(5):
        start = time.perf_counter()
        _ = llm.invoke(messages)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    print(f"LLM Inference (5 runs): {avg_time:.2f}s avg")
    return avg_time


def benchmark_summarization():
    """Benchmark summarization speed."""
    # Create a test document ~5000 tokens
    test_text = (
        "This is a test document for summarization. " * 1000
    )  # ~5000 tokens
    
    # Warmup
    _ = summarize_text(test_text[:1000])
    
    # Benchmark single-pass
    start = time.perf_counter()
    result = summarize_text(test_text)
    elapsed = time.perf_counter() - start
    
    print(f"Summarization (~5000 tokens): {elapsed:.2f}s")
    print(f"  Result length: {len(result)} chars")
    return elapsed


def benchmark_embeddings():
    """Benchmark embedding speed."""
    embeddings = get_embeddings()
    texts = [f"Test document number {i} with some content to embed." for i in range(100)]
    
    # Warmup
    _ = embeddings.embed_documents(texts[:10])
    
    # Benchmark
    start = time.perf_counter()
    _ = embeddings.embed_documents(texts)
    elapsed = time.perf_counter() - start
    
    print(f"Embeddings (100 docs): {elapsed:.2f}s ({100/elapsed:.1f} docs/s)")
    return elapsed


def benchmark_agent_query():
    """Benchmark full agent query."""
    query = "What is machine learning?"
    messages = [HumanMessage(content=query)]
    
    # Warmup
    _ = run_agent(messages)
    
    # Benchmark
    start = time.perf_counter()
    result = run_agent(messages)
    elapsed = time.perf_counter() - start
    
    print(f"Agent Query (simple): {elapsed:.2f}s")
    return elapsed


def benchmark_agent_complex():
    """Benchmark complex agent query with tools."""
    query = "Search for recent AI research papers and summarize key findings."
    messages = [HumanMessage(content=query)]
    
    start = time.perf_counter()
    result = run_agent(messages)
    elapsed = time.perf_counter() - start
    
    print(f"Agent Query (with tools): {elapsed:.2f}s")
    return elapsed


def main():
    print("=" * 60)
    print("Performance Benchmark - Local AI Research Assistant")
    print("=" * 60)
    print(f"Backend: {settings.llm_backend}")
    if settings.llm_backend == "vllm":
        print(f"Model: {settings.vllm_model}")
        print(f"Quantization: {settings.vllm_quantization}")
        print(f"Prefix Caching: {settings.vllm_enable_prefix_caching}")
    print(f"Embeddings: {settings.embedding_model} (ONNX: {settings.embedding_use_onnx})")
    print("=" * 60)
    
    results = {}
    
    print("\n1. LLM Inference Benchmark...")
    results["llm"] = benchmark_llm_inference()
    
    print("\n2. Summarization Benchmark...")
    results["summarize"] = benchmark_summarization()
    
    print("\n3. Embeddings Benchmark...")
    results["embeddings"] = benchmark_embeddings()
    
    print("\n4. Agent Query (Simple)...")
    results["agent_simple"] = benchmark_agent_query()
    
    print("\n5. Agent Query (Complex with Tools)...")
    results["agent_complex"] = benchmark_agent_complex()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, time_val in results.items():
        print(f"  {name}: {time_val:.2f}s")


if __name__ == "__main__":
    main()