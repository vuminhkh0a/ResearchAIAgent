# Performance Optimizations Summary

## Changes Made

### 1. LLM Backend: Ollama → vLLM (Highest Impact)
**File:** `llm_vllm.py`, `llm.py`, `config/settings.py`

| Aspect | Before (Ollama) | After (vLLM) |
|--------|-----------------|--------------|
| Model | qwen3:8b (8B params) | Qwen/Qwen3-4B-Instruct (AWQ 4-bit) |
| Quantization | None (FP16) | AWQ 4-bit |
| KV-Cache | No reuse | Prefix caching enabled |
| Batching | Sequential | Continuous batching (4096 tokens) |
| Compilation | None | CUDA graphs (`enforce_eager=False`) |
| Expected Speedup | Baseline | **3-5x faster** |

**Key Settings for RTX 3060/4060 (8-12GB VRAM):**
```python
VLLM_MODEL=Qwen/Qwen3-4B-Instruct
VLLM_QUANTIZATION=awq
VLLM_GPU_MEMORY_UTILIZATION=0.85
VLLM_ENABLE_PREFIX_CACHING=true
VLLM_ENFORCE_EAGER=false  # CUDA graphs
VLLM_MAX_NUM_BATCHED_TOKENS=4096
```

---

### 2. Summarization: Map-Reduce → Single-Pass + Hierarchical
**File:** `summarization/map_reduce.py`

| Approach | LLM Calls | Time (est.) |
|----------|-----------|-------------|
| Map-Reduce (old) | 10-20+ | 60-120s |
| Single-pass (new) | 1 | 3-5s |
| Hierarchical fallback | 2-3 | 6-15s |

**Logic:**
- If document fits in context (≤7000 tokens): single pass
- If longer: split into chunks → summarize each → merge (2-3 calls max)
- **Eliminates recursive map-reduce loops**

---

### 3. Agent ReAct Loop: Unlimited → Max 2 Rounds
**File:** `agent/graph.py`, `agent/prompts.py`, `agent/state.py`

| Aspect | Before | After |
|--------|--------|-------|
| Max tool rounds | Unlimited | 2 |
| Tool calling | Sequential | Parallel encouraged |
| System prompt | Basic | Explicit parallel tool rules |
| Streaming | No | Yes (perceived latency) |

**Prompt additions:**
```
TOOL CALLING RULES:
1. Call MULTIPLE tools in a SINGLE response when needed
2. Prefer parallel tool calls over sequential
3. Max 2 rounds of tool calls per query
```

---

### 4. Embeddings: CPU → ONNX Runtime
**File:** `rag/embeddings.py`

| Aspect | Before | After |
|--------|--------|-------|
| Runtime | PyTorch CPU | ONNX Runtime CPU |
| Batching | No | Yes (batch_size=32) |
| Expected Speedup | Baseline | **3-5x faster** |

---

### 5. Chunking: Smaller → Larger (Fewer Retrievals)
**File:** `config/settings.py`

| Setting | Before | After |
|---------|--------|-------|
| CHUNK_SIZE | 800 | 1500 |
| CHUNK_OVERLAP | 120 | 200 |
| MAX_CONTEXT_TOKENS | 2500 | 4000 |
| RESERVED_RESPONSE_TOKENS | 1024 | 2048 |

Larger chunks = fewer retrievals = fewer LLM calls.

---

### 6. Streaming Responses
**File:** `llm_vllm.py`, `app.py`, `agent/graph.py`

- Token-by-token streaming via vLLM generator
- Perceived latency improvement (user sees output immediately)
- `run_agent_stream()` for real-time UI updates

---

## Expected Performance Improvements

| Task | Before | After | Improvement |
|------|--------|-------|-------------|
| Simple QA | 15-30s | 3-5s | **5-6x** |
| Summarization (5k tokens) | 60-120s | 3-8s | **10-15x** |
| Document ingestion (100 chunks) | 30-60s | 6-15s | **4-5x** |
| Complex query (web + RAG) | 45-90s | 8-15s | **5-6x** |

---

## Trade-offs

| Optimization | Trade-off |
|--------------|-----------|
| 4B model vs 8B | Slightly lower quality on complex reasoning |
| Single-pass summary | May miss details in very long docs (mitigated by hierarchical fallback) |
| Max 2 tool rounds | May not handle extremely complex multi-step queries |
| AWQ quantization | ~1-2% quality loss vs FP16 |
| ONNX embeddings | First run slower (export), then faster |
| CUDA graphs | First inference slower (compilation), then faster |

---

## Installation

```bash
# Install pinned requirements
pip install -r requirements.txt

# Download model (first run)
# vLLM will auto-download Qwen/Qwen3-4B-Instruct AWQ model
```

---

## Profiling & Monitoring

Run benchmarks:
```bash
python benchmark.py
```

Key metrics to monitor:
- `llm` - raw inference latency
- `summarize` - summarization speed
- `embeddings` - embedding throughput (docs/sec)
- `agent_simple` - end-to-end simple query
- `agent_complex` - end-to-end with tools

---

## Reverting to Ollama

If vLLM has issues, set in `.env`:
```
LLM_BACKEND=ollama
OLLAMA_MODEL=qwen3:8b
```