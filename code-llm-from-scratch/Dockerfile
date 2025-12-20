# Dockerfile for Code LLM API Deployment
#
# Build: docker build -t code-llm-api .
# Run:   docker run -p 8000:8000 code-llm-api
#
# Multi-stage build for smaller image size

# Stage 1: Build stage
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Install FastAPI and uvicorn
RUN pip install --no-cache-dir --user fastapi uvicorn[standard] pydantic

# Stage 2: Runtime stage
FROM python:3.10-slim

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/ src/
COPY examples/ examples/
COPY models/ models/

# Expose port
EXPOSE 8000

# Environment variables
ENV MODEL_PATH=models/code/code_model_final.pt
ENV TOKENIZER_PATH=models/language/language_tokenizer.json
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the API
CMD ["uvicorn", "examples.deployment_api:app", "--host", "0.0.0.0", "--port", "8000"]
