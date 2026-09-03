# Multi-arch image: linux/amd64 (Windows/Linux Intel/AMD, macOS Intel)
# and linux/arm64 (macOS Apple Silicon, Linux ARM, e.g. Raspberry Pi).
# Windows users run this as a Linux container under Docker Desktop (default mode).
FROM python:3.11-slim-bookworm

# ---- System dependencies ----
# curl: container HEALTHCHECK; libgomp1: required by onnxruntime (embeddings).
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Python environment ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Hugging Face / ONNX cache lives here (mount a volume to persist it)
    HF_HOME=/home/appuser/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    # Streamlit: headless + no usage stats inside containers
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install Python deps first (better layer caching: rebuilds are fast
# unless requirements.txt changes).
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---- App source ----
COPY app.py check_setup.py benchmark.py llm.py llm_vllm.py .env.example ./
COPY agent/ ./agent/
COPY config/ ./config/
COPY loaders/ ./loaders/
COPY rag/ ./rag/
COPY summarization/ ./summarization/
COPY tools/ ./tools/
COPY utils/ ./utils/
COPY .streamlit/ ./.streamlit/

# Data dirs (Chroma DB, uploads, ingestion index). Mount a volume here
# in compose/K8s so documents survive container restarts.
RUN mkdir -p data/chroma data/uploads

# Run as non-root user.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app /home/appuser
USER appuser

# Optional: bake the ~90 MB embedding model into the image so the first
# run doesn't download it. Comment out to keep the image smaller / for
# fully offline builds you manage yourself.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Bind to 0.0.0.0 so the UI is reachable outside the container.
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
