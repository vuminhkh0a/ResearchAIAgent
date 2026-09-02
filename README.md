# Local AI Research Assistant

A lightweight research assistant running **Qwen3 locally via Ollama** with GPU acceleration. LangGraph orchestrates the agent; RAG, web search, calculator, and summarization are tools the agent invokes dynamically.

## Features

- Local LLM inference with **Ollama + Qwen3** (GPU-accelerated, no cloud APIs)
- Streamlit chat UI with file upload (PDF, DOCX, TXT, HTML)
- RAG with persistent **Chroma** vector store
- Dynamic tool calling: `file_search`, `web_search`, `calculator`, `summarize`
- Single-pass + hierarchical fallback summarization (1-3 LLM calls vs 10-20)
- Source citations (filename/page for RAG; title/URL for web search)
- ONNX-optimized embeddings (3-5x faster)
- Streaming responses for perceived latency improvement

## Quick Start (3 terminals)

### Terminal 1: Start Ollama
```bash
# Windows PowerShell
ollama serve

# macOS / Linux
ollama serve
```
Keep this running. Ollama loads models on-demand to GPU.

### Terminal 2: Pull Model (one-time)
```bash
# Windows PowerShell
ollama pull qwen3:4b

# macOS / Linux
ollama pull qwen3:4b
```
`qwen3:4b` (2.5 GB) fits in 6GB VRAM. For larger GPUs: `qwen3:8b` (5.2 GB).

### Terminal 3: Run App
```bash
# Windows PowerShell (from project root)
cd C:\Users\LENOVO\ResearchAiAgent
.\.venv\Scripts\python.exe -m streamlit run app.py

# macOS / Linux (from project root)
cd /path/to/ResearchAiAgent
.venv/bin/python -m streamlit run app.py
```
Open `http://localhost:8501` in browser.

---

## Full Installation

### 1. Clone & Create Virtual Environment

**Windows PowerShell:**
```powershell
cd C:\Users\LENOVO\ResearchAiAgent
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
```

**macOS / Linux:**
```bash
cd /path/to/ResearchAiAgent
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

### 2. Install Ollama

- **Windows/macOS:** Download from [ollama.com](https://ollama.com)
- **Linux:** `curl -fsSL https://ollama.com/install.sh | sh`

### 3. Pull Model & Verify

```bash
# Pull 4B model (recommended for 6-8GB VRAM)
ollama pull qwen3:4b

# Or 8B model (needs 8GB+ VRAM)
ollama pull qwen3:8b

# Verify
ollama list
ollama run qwen3:4b "Reply: Ollama working."
```

### 4. Configure `.env`

Edit `.env` (copied from `.env.example`):

```env
# LLM Backend: "ollama" (default, cross-platform) or "vllm" (Linux only, faster)
LLM_BACKEND=ollama

# Ollama model (must match `ollama list`)
OLLAMA_MODEL=qwen3:4b

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_USE_ONNX=true

# RAG settings
CHUNK_SIZE=1500
CHUNK_OVERLAP=200
RETRIEVAL_K=4
MAX_CONTEXT_TOKENS=4000
```

---

## GPU Verification

**Check GPU is used by Ollama:**
```bash
# In another terminal while querying
nvidia-smi -l 1
```
You should see `ollama` process using GPU memory when answering.

**Test inference speed:**
```bash
# Quick test
python -c "
from llm import get_llm
from langchain_core.messages import HumanMessage
import time
llm = get_llm()
start = time.perf_counter()
result = llm.invoke([HumanMessage(content='Explain quantum computing in 2 sentences.')])
print(f'Time: {time.perf_counter()-start:.2f}s')
print(f'Response: {result.content[:100]}')
"
```

---

## Architecture

```
User → Streamlit → LangGraph Agent → Qwen3 (Ollama)
                         ↓
              ┌────────┴────────┐
              ↓                 ↓
         Tools (parallel)   Final Answer
              ↓
         ┌────┴────┐
         ↓         ↓
    file_search  web_search
    calculator   summarize
```

**Optimizations applied:**
- Max 2 tool-calling rounds (prevents infinite loops)
- Parallel tool execution encouraged in system prompt
- Single-pass summarization for docs ≤ 7000 tokens
- Hierarchical fallback (2-3 calls max) for long docs
- ONNX Runtime for embeddings (batched, 32 docs/call)
- Larger chunks (1500 tokens) = fewer retrievals

---

## Environment Variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | `ollama` | `ollama` (cross-platform) or `vllm` (Linux only) |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server |
| `OLLAMA_MODEL` | `qwen3:4b` | Must match `ollama list` |
| `VLLM_MODEL` | `Qwen/Qwen3-4B-Instruct` | vLLM model (Linux only) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embeddings |
| `EMBEDDING_USE_ONNX` | `true` | ONNX Runtime optimization |
| `CHUNK_SIZE` | `1500` | Text chunk size |
| `CHUNK_OVERLAP` | `200` | Chunk overlap |
| `RETRIEVAL_K` | `4` | Top-k chunks |
| `MAX_CONTEXT_TOKENS` | `4000` | Context budget |
| `SEARCH_PROVIDER` | `duckduckgo` | `duckduckgo` or `tavily` |
| `TAVILY_API_KEY` | `` | Required for Tavily |
| `LLM_TEMPERATURE` | `0.1` | Generation temperature |

---

## Example Queries

| Task | Query |
|------|-------|
| General Q&A | `What is retrieval-augmented generation?` |
| RAG | Upload paper → `What is the main contribution?` |
| Web search | `Recent LoRA approaches with sources` |
| Calculation | `What is 1234 * 0.15?` |
| Summarize | `Summarize the uploaded document` |
| Multi-tool | `Search web for LoRA papers and compare with uploaded PDF` |

---

## Project Structure

```
ResearchAiAgent/
├── app.py                 # Streamlit UI
├── llm.py                 # LLM factory (auto-fallback: vLLM → Ollama)
├── llm_vllm.py            # vLLM wrapper (Linux only)
├── requirements.txt       # Pinned dependencies
├── .env.example           # Template config
├── .env                   # Your config (gitignored)
├── config/settings.py     # Settings loader
├── agent/
│   ├── graph.py           # LangGraph ReAct (max 2 rounds, streaming)
│   ├── state.py           # AgentState with tool_rounds
│   └── prompts.py         # System prompt with parallel tool rules
├── tools/
│   ├── file_search.py     # RAG retrieval
│   ├── web_search.py      # DuckDuckGo / Tavily
│   ├── calculator.py      # Safe math eval
│   └── summarize.py       # Summarization tool
├── rag/
│   ├── embeddings.py      # ONNX-optimized embeddings
│   ├── ingestion.py       # File → Chroma
│   ├── retriever.py       # Token-budgeted retrieval
│   └── vector_store.py    # Chroma wrapper
├── summarization/
│   └── map_reduce.py      # Single-pass + hierarchical
├── loaders/
│   └── document_loader.py # PDF/DOCX/TXT/HTML
└── utils/
    ├── tokens.py          # Token estimation
    ├── ollama.py          # Ollama health checks
    └── errors.py          # User-facing errors
```

---

## Performance Notes

| Task | Typical Time (qwen3:4b, RTX 4050) |
|------|-----------------------------------|
| Simple Q&A | 2-4s |
| Summarize 5K tokens | 3-6s (single-pass) |
| Summarize 20K tokens | 8-15s (hierarchical) |
| Web search + answer | 4-8s |
| RAG query | 3-5s |
| Embeddings (100 docs) | 1-2s |

---

## For Linux Users: vLLM (Faster)

```bash
# Requires CUDA 12.1+, Linux
pip install vllm==0.6.0
# Edit .env:
LLM_BACKEND=vllm
VLLM_MODEL=Qwen/Qwen3-4B-Instruct
```

vLLM provides: continuous batching, prefix caching (KV-cache reuse), CUDA graphs.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ollama: command not found` | Install Ollama, restart terminal |
| `Model not found` | `ollama pull qwen3:4b` |
| GPU not used | Update NVIDIA drivers, check `nvidia-smi` |
| Slow first query | First query loads model to VRAM (~5-10s) |
| Port 8501 in use | `streamlit run app.py --server.port 8502` |
| Execution policy blocked | Use `.\.venv\Scripts\python.exe -m streamlit run app.py` directly |

---

## License

MIT