# Document Intelligence Refinery
# Multi-stage build: slim runtime image with all dependencies pre-installed.
#
# Build:  docker build -t refinery .
# Run:    docker run --env-file .env -v $(pwd)/data:/app/data -v $(pwd)/.refinery:/app/.refinery refinery
# Query:  docker run --env-file .env refinery python -m src.agents.query_agent "What was net profit?" <doc_id>

FROM python:3.11-slim AS base

# System deps: poppler (for pdfplumber image extraction), libmagic (file type detection)
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        libmagic1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── dependency layer (cached unless pyproject.toml changes) ──────────────────
COPY pyproject.toml ./

# Install core + final dependencies (excludes docling heavy model downloads)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        pdfplumber>=0.11 \
        pymupdf>=1.24 \
        pydantic>=2.0 \
        pyyaml>=6.0 \
        python-dotenv>=1.0 \
        tiktoken>=0.7 \
        httpx>=0.27 \
        rich>=13.0 \
        chromadb>=0.5 \
        sentence-transformers>=3.0

# ── application layer ────────────────────────────────────────────────────────
COPY src/ ./src/
COPY rubric/ ./rubric/
COPY examples/ ./examples/
COPY scripts/ ./scripts/

# Pre-download the sentence-transformers embedding model so runtime is air-gapped
RUN python3 -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()" || true

# Create persistent directories
RUN mkdir -p .refinery/profiles .refinery/pageindex .refinery/chroma

# ── runtime configuration ────────────────────────────────────────────────────
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check — verifies the store is importable
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "from src.store.vector_store import VectorStore; print('ok')" || exit 1

# Default command: interactive Q&A
CMD ["python3", "-m", "src.agents.query_agent", "--help"]
