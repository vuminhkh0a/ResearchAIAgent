# Local AI Research Assistant

A lightweight research assistant running **Qwen3 locally via Ollama** with GPU acceleration. LangGraph orchestrates the agent; RAG, web search, calculator, and summarization are tools the agent invokes dynamically.

Runs on **Windows, Linux, and macOS** — with **Docker** (easiest), **Kubernetes**, or native **Python**.

## Features

- Local LLM inference with **Ollama + Qwen3** (GPU-accelerated, no cloud APIs)
- Streamlit chat UI with file upload (PDF, DOCX, TXT, HTML)
- RAG with persistent **Chroma** vector store
- Dynamic tool calling: `file_search`, `web_search`, `calculator`, `summarize`
- Single-pass + hierarchical fallback summarization (1-3 LLM calls vs 10-20)
- Source citations (filename/page for RAG; title/URL for web search)
- ONNX-optimized embeddings (3-5x faster)
- Streaming responses for perceived latency improvement
- Multi-arch **Docker** image (`amd64` + `arm64`: Intel/AMD PCs, Apple Silicon Macs)
- **Kubernetes** manifests (namespace, ConfigMap, PVCs, Ollama + app, model-pull Job)

## Contents

- [INSTALLATION](#installation)
  - [Windows users](#windows-users)
  - [macOS users](#macos-users)
  - [Linux users](#linux-users)
  - [Kubernetes (any system, optional)](#kubernetes-any-system-optional)
- [Settings](#settings)
- [Architecture](#architecture)
- [Example Queries](#example-queries)
- [Project Structure](#project-structure)
- [Performance Notes](#performance-notes)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## INSTALLATION

Find yourself below. Every path ends with the app at **`http://localhost:8501`**.

### Windows users

Recommended: Docker (no Python setup needed — app, Ollama, and the model come in containers).

```powershell
winget install Docker.DockerDesktop
```

Start **Docker Desktop** from the Start menu and wait for "Engine running". Then:

```powershell
cd path\to\ResearchAiAgent
docker compose up --build -d
```

First start takes a few minutes (the ~2.5 GB `qwen3:4b` model downloads automatically). Open `http://localhost:8501`. Done.

- NVIDIA GPU: Docker Desktop already uses the WSL2 backend — just run with the GPU override instead: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d`
- No Docker? Install Python 3.11+ and [Ollama](https://ollama.com), then: `python -m venv .venv` → `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` → `copy .env.example .env` → `ollama pull qwen3:4b` → `.\.venv\Scripts\python.exe -m streamlit run app.py`
- Stop anytime: `docker compose down` (data kept) · logs: `docker compose logs -f app`

### macOS users

Recommended: Docker (works on Intel and Apple Silicon; no Python setup needed).

```bash
brew install --cask docker
```

Open **Docker Desktop** from Applications and wait for "Engine running". Then:

```bash
cd /path/to/ResearchAiAgent
docker compose up --build -d
```

First start takes a few minutes (the ~2.5 GB `qwen3:4b` model downloads automatically). Open `http://localhost:8501`. Done.

- Note: containers can't use the GPU on Mac, so inference is CPU-only (~3-10x slower than NVIDIA). For faster answers, skip Docker: install Python 3.11+ and [Ollama for Mac](https://ollama.com) (uses Metal), then: `python3 -m venv .venv` → `.venv/bin/python -m pip install -r requirements.txt` → `cp .env.example .env` → `ollama pull qwen3:4b` → `.venv/bin/python -m streamlit run app.py`
- Stop anytime: `docker compose down` (data kept) · logs: `docker compose logs -f app`

### Linux users

Recommended: Docker (no Python setup needed — app, Ollama, and the model come in containers).

```bash
curl -fsSL https://get.docker.com | sh
```

(Log out and back in afterwards so `docker` runs without `sudo`.) Then:

```bash
cd /path/to/ResearchAiAgent
docker compose up --build -d
```

First start takes a few minutes (the ~2.5 GB `qwen3:4b` model downloads automatically). Open `http://localhost:8501`. Done.

- NVIDIA GPU: install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), then run with the GPU override instead: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d`
- No Docker? Install Python 3.11+ and Ollama (`curl -fsSL https://ollama.com/install.sh | sh`), then: `python3 -m venv .venv` → `.venv/bin/python -m pip install -r requirements.txt` → `cp .env.example .env` → `ollama pull qwen3:4b` → `.venv/bin/python -m streamlit run app.py`
- Maximum speed (native only, CUDA 12.1+): `pip install vllm==0.6.0`, then set `LLM_BACKEND=vllm` in `.env`
- Stop anytime: `docker compose down` (data kept) · logs: `docker compose logs -f app`

### Kubernetes (any system, optional)

For clusters (Docker Desktop K8s, minikube, kind, cloud) instead of Docker Compose.

```bash
# 1. Install kubectl (+ a cluster if needed):
#    Windows: winget install Kubernetes.kubectl Kubernetes.minikube
#    macOS:   brew install kubectl minikube
#    Linux:   curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && sudo install kubectl /usr/local/bin/kubectl
#    Then (minikube only): minikube start --driver=docker

# 2. Build the image and give it to your cluster:
cd /path/to/ResearchAiAgent
docker build -t research-agent:latest .
kind load docker-image research-agent:latest      # kind only
minikube image load research-agent:latest         # minikube only
# Docker Desktop K8s needs neither load command; cloud clusters: push to a
# registry and update `image:` in k8s/app.yaml instead.

# 3. Deploy everything (app + Ollama + model download):
kubectl apply -k k8s/

# 4. Open the app:
kubectl -n research-agent port-forward svc/app 8501:8501
# → http://localhost:8501. Done.
```

---

## Settings

Defaults work out of the box. Change them where your deployment reads config:

| Deployment | Where to change |
|------------|-----------------|
| Docker | `.env` file or shell exports (`OLLAMA_MODEL`, `RETRIEVAL_K`, `TAVILY_API_KEY`, …). `OLLAMA_BASE_URL`/`LLM_BACKEND` are pinned in `docker-compose.yml` — don't change them |
| Kubernetes | `k8s/configmap.yaml`, then `kubectl apply -k k8s/` + `kubectl -n research-agent rollout restart deploy/app`. Optional secret: `kubectl -n research-agent create secret generic app-secrets --from-literal=TAVILY_API_KEY=tvly-...` |
| Native | `.env` file |

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | `ollama` | `ollama` (all platforms, Docker, K8s) or `vllm` (native Linux only) |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server (auto-set to `http://ollama:11434` in Docker/K8s) |
| `OLLAMA_MODEL` | `qwen3:4b` | Model name — must be the one downloaded (`qwen3:4b` ~2.5 GB, `qwen3:8b` ~5.2 GB) |
| `VLLM_MODEL` | `Qwen/Qwen3-4B-Instruct` | vLLM model (native Linux only) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embeddings |
| `EMBEDDING_USE_ONNX` | `true` | ONNX Runtime optimization |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1500` / `200` | RAG chunking |
| `RETRIEVAL_K` | `4` | Top-k chunks |
| `MAX_CONTEXT_TOKENS` | `4000` | Context budget |
| `SEARCH_PROVIDER` | `duckduckgo` | `duckduckgo` (no key) or `tavily` (needs `TAVILY_API_KEY`) |
| `LLM_TEMPERATURE` | `0.1` | Generation temperature |

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

Containerized:

```
Docker:  [app :8501] ⇄ [ollama :11434] + auto model-pull   volumes: app-data, hf-cache, ollama-data
K8s:     svc/app → deploy/app (PVC app-data, ConfigMap app-config)
         svc/ollama → deploy/ollama (PVC ollama-models) + model-pull Job (auto-applied)
```

**Optimizations applied:**
- Max 2 tool-calling rounds (prevents infinite loops)
- Parallel tool execution encouraged in system prompt
- Single-pass summarization for docs ≤ 7000 tokens
- Hierarchical fallback (2-3 calls max) for long docs
- ONNX Runtime for embeddings (batched, 32 docs/call)
- Larger chunks (1500 tokens) = fewer retrievals

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
├── app.py                      # Streamlit UI
├── llm.py                      # LLM factory (auto-fallback: vLLM → Ollama)
├── llm_vllm.py                 # vLLM wrapper (native Linux only)
├── requirements.txt            # Pinned dependencies
├── Dockerfile                  # Multi-arch app image (amd64/arm64)
├── .dockerignore               # Keeps secrets/data out of the image
├── docker-compose.yml          # App + Ollama + auto model-pull (all OSes)
├── docker-compose.gpu.yml      # GPU override (Linux / Windows-WSL2 + NVIDIA)
├── k8s/
│   ├── kustomization.yaml      # `kubectl apply -k k8s/` (includes model-pull Job)
│   ├── namespace.yaml          # research-agent namespace
│   ├── configmap.yaml          # non-secret config
│   ├── pvc.yaml                # app-data + ollama-models volumes
│   ├── ollama.yaml             # Ollama Deployment + Service (+GPU notes)
│   ├── app.yaml                # App Deployment + Service (1 replica)
│   └── model-pull-job.yaml     # one-time model download Job
├── .env.example                # Template config
├── .env                        # Your config (gitignored)
├── config/settings.py          # Settings loader
├── agent/
│   ├── graph.py                # LangGraph ReAct (max 2 rounds, streaming)
│   ├── state.py                # AgentState with tool_rounds
│   └── prompts.py              # System prompt with parallel tool rules
├── tools/
│   ├── file_search.py          # RAG retrieval
│   ├── web_search.py           # DuckDuckGo / Tavily
│   ├── calculator.py           # Safe math eval
│   └── summarize.py            # Summarization tool
├── rag/
│   ├── embeddings.py           # ONNX-optimized embeddings
│   ├── ingestion.py            # File → Chroma
│   ├── retriever.py            # Token-budgeted retrieval
│   └── vector_store.py         # Chroma wrapper
├── summarization/
│   └── map_reduce.py           # Single-pass + hierarchical
├── loaders/
│   └── document_loader.py      # PDF/DOCX/TXT/HTML
└── utils/
    ├── tokens.py               # Token estimation
    ├── ollama.py               # Ollama health checks
    └── errors.py               # User-facing errors
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

Docker/Kubernetes add negligible overhead. On macOS (CPU-only Ollama in Docker) expect roughly 3-10x slower inference than an NVIDIA GPU — for faster Mac inference, run natively with Ollama for Mac (Metal acceleration) instead.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `docker: command not found` | Install Docker, then restart your terminal |
| "Ollama is not available" / "model not found" (Docker) | Wait for first-start download to finish: `docker compose logs -f pull-model`; or manually: `docker exec research-ollama ollama pull qwen3:4b` |
| "Ollama is not available" (K8s) | `kubectl -n research-agent get pods`; check download: `kubectl -n research-agent logs job/ollama-model-pull` |
| Port 8501 in use | `docker compose down`, or native: `streamlit run app.py --server.port 8502` |
| `ollama: command not found` (native) | Install Ollama, restart terminal |
| Slow first query | First query loads the model (~5-10s) — normal |
| GPU not used (Docker) | Linux: NVIDIA Container Toolkit + `docker-compose.gpu.yml`; check `nvidia-smi` |
| K8s `ImagePullBackOff` | Build locally (`docker build -t research-agent:latest .`), load into kind/minikube, or push to a registry and update `k8s/app.yaml` |
| K8s PVC stuck Pending | Needs a default StorageClass (Docker Desktop/minikube include one) |
| Execution policy blocked (Windows native) | Use `.\.venv\Scripts\python.exe -m streamlit run app.py` directly |

---

## License

MIT
