# Dockerfile for Code Generation API
# Multi-stage build for smaller production image

# ============================================================
# Stage 1: Builder
# ============================================================
FROM python:3.10-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ============================================================
# Stage 2: Production
# ============================================================
FROM python:3.10-slim as production

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/ src/
COPY scripts/ scripts/

# Copy tokenizer (small, can be in image)
COPY data/tokenizer/ data/tokenizer/

# Model files should be mounted or downloaded at runtime
# COPY models/ models/

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/code_model_best.pth \
    TOKENIZER_PATH=/app/data/tokenizer/tokenizer.json \
    PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API server
CMD ["python", "-m", "uvicorn", "scripts.serve_api:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================================
# Stage 3: Development (optional)
# ============================================================
FROM production as development

USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    httpx \
    black \
    flake8 \
    mypy

# Copy test files
COPY tests/ tests/

USER appuser

CMD ["python", "-m", "pytest", "tests/", "-v"]

# ============================================================
# Stage 4: GPU Production (for CUDA inference)
# ============================================================
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04 as gpu-production

WORKDIR /app

# Install Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Update pip
RUN python3 -m pip install --upgrade pip

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY scripts/ scripts/
COPY data/tokenizer/ data/tokenizer/

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/code_model_best.pth \
    TOKENIZER_PATH=/app/data/tokenizer/tokenizer.json \
    DEVICE=cuda \
    PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API server
CMD ["python3", "-m", "uvicorn", "scripts.serve_api:app", "--host", "0.0.0.0", "--port", "8000"]
