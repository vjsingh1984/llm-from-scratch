#!/usr/bin/env python3
"""
FastAPI Server for Code Generation
Production-ready API for serving the code generation model
"""

import os
import sys
import time
import asyncio
import logging
from typing import Optional
from collections import defaultdict
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, HTTPException, Depends, Security, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
import torch
import hashlib

# ============================================================
# Configuration
# ============================================================

class Config:
    MODEL_PATH = os.environ.get("MODEL_PATH", "models/code_model_best.pth")
    TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "data/tokenizer/tokenizer.json")
    MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))
    DEVICE = os.environ.get("DEVICE", "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT", "60"))
    API_KEYS_ENABLED = os.environ.get("API_KEYS_ENABLED", "false").lower() == "true"

# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger("code_gen_api")

# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Code Generation API",
    description="Production API for code generation using transformer models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Request/Response Models
# ============================================================

class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="The code prompt to complete", min_length=1, max_length=4096)
    max_tokens: int = Field(256, description="Maximum tokens to generate", ge=1, le=2048)
    temperature: float = Field(0.7, description="Sampling temperature", ge=0.0, le=2.0)
    top_p: float = Field(0.9, description="Top-p sampling parameter", ge=0.0, le=1.0)
    top_k: int = Field(50, description="Top-k sampling parameter", ge=1, le=100)
    language: Optional[str] = Field(None, description="Target programming language")
    stop_sequences: Optional[list] = Field(None, description="Stop generation at these sequences")

class GenerationResponse(BaseModel):
    generated_code: str
    tokens_generated: int
    generation_time_ms: float
    model_version: str
    finish_reason: str  # "length", "stop", "error"

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    uptime_seconds: float

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

# ============================================================
# Model Loading
# ============================================================

class ModelManager:
    """Manages model loading and inference"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.loaded = False
        self.start_time = time.time()

    def load(self):
        """Load model and tokenizer"""
        try:
            from tokenizers import Tokenizer
            from src.model import create_model
            from src.model.config import ModelConfig

            logger.info(f"Loading model from {Config.MODEL_PATH}")

            # Load tokenizer
            self.tokenizer = Tokenizer.from_file(Config.TOKENIZER_PATH)

            # Create model config
            self.config = ModelConfig(
                vocab_size=self.tokenizer.get_vocab_size(),
                hidden_size=1024,
                num_layers=24,
                num_heads=16,
                intermediate_size=4096,
                max_position_embeddings=2048,
                use_rmsnorm=True,
                use_rope=True,
            )

            # Create and load model
            self.model = create_model(self.config, device=Config.DEVICE)

            checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.eval()
            self.loaded = True
            logger.info(f"Model loaded successfully on {Config.DEVICE}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    @torch.no_grad()
    def generate(self, prompt: str, max_tokens: int = 256,
                 temperature: float = 0.7, top_p: float = 0.9,
                 top_k: int = 50, stop_sequences: list = None) -> tuple:
        """Generate code from prompt"""

        if not self.loaded:
            raise RuntimeError("Model not loaded")

        # Tokenize
        encoded = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], device=Config.DEVICE)

        # Generate
        generated_ids = input_ids.clone()
        finish_reason = "length"

        for _ in range(max_tokens):
            # Get logits
            outputs = self.model(generated_ids)
            next_token_logits = outputs[:, -1, :] / max(temperature, 1e-7)

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Check for EOS
            if next_token.item() == self.tokenizer.token_to_id("[EOS]"):
                finish_reason = "stop"
                break

            # Check for stop sequences
            if stop_sequences:
                current_text = self.tokenizer.decode(generated_ids[0].tolist())
                for stop in stop_sequences:
                    if stop in current_text[len(prompt):]:
                        finish_reason = "stop"
                        break
                if finish_reason == "stop":
                    break

        # Decode
        generated_text = self.tokenizer.decode(generated_ids[0].tolist())
        new_text = generated_text[len(prompt):]
        tokens_generated = len(generated_ids[0]) - len(input_ids[0])

        return new_text, tokens_generated, finish_reason

# Global model manager
model_manager = ModelManager()

# ============================================================
# Rate Limiting
# ============================================================

request_counts = defaultdict(list)

def get_client_id(request) -> str:
    """Get client identifier for rate limiting"""
    # In production, use proper client identification
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0]
    return request.client.host if request.client else "unknown"

async def check_rate_limit(client_id: str):
    """Check if client has exceeded rate limit"""
    now = time.time()
    # Clean old requests
    request_counts[client_id] = [t for t in request_counts[client_id] if now - t < 60]

    if len(request_counts[client_id]) >= Config.RATE_LIMIT_PER_MINUTE:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {Config.RATE_LIMIT_PER_MINUTE} requests per minute."
        )

    request_counts[client_id].append(now)

# ============================================================
# API Key Authentication (Optional)
# ============================================================

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# In production, use a proper secrets manager
VALID_API_KEYS = {
    hashlib.sha256(b"dev_test_key").hexdigest(): "Development",
    # Add production keys here
}

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key if authentication is enabled"""
    if not Config.API_KEYS_ENABLED:
        return None

    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    if key_hash not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return VALID_API_KEYS[key_hash]

# ============================================================
# Startup/Shutdown Events
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        model_manager.load()
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        # Continue without model - will return errors on generate requests

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API server")

# ============================================================
# API Endpoints
# ============================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancers"""
    return HealthResponse(
        status="healthy" if model_manager.loaded else "degraded",
        model_loaded=model_manager.loaded,
        device=Config.DEVICE,
        uptime_seconds=time.time() - model_manager.start_time
    )

@app.get("/ready")
async def readiness_check():
    """Readiness probe for Kubernetes"""
    if not model_manager.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"ready": True}

@app.post("/generate", response_model=GenerationResponse)
async def generate_code(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    api_key_info: str = Depends(verify_api_key)
):
    """Generate code from a prompt"""

    # Check model is loaded
    if not model_manager.loaded:
        raise HTTPException(status_code=503, detail="Model not available")

    start_time = time.time()

    try:
        # Generate
        generated_code, tokens_generated, finish_reason = await asyncio.to_thread(
            model_manager.generate,
            request.prompt,
            request.max_tokens,
            request.temperature,
            request.top_p,
            request.top_k,
            request.stop_sequences
        )

        generation_time = (time.time() - start_time) * 1000

        # Log request
        logger.info(f"Generated {tokens_generated} tokens in {generation_time:.2f}ms")

        return GenerationResponse(
            generated_code=generated_code,
            tokens_generated=tokens_generated,
            generation_time_ms=generation_time,
            model_version="1.0.0",
            finish_reason=finish_reason
        )

    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-generate")
async def batch_generate(
    requests: list[GenerationRequest],
    api_key_info: str = Depends(verify_api_key)
):
    """Generate code for multiple prompts (batch processing)"""

    if len(requests) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 requests per batch")

    results = []
    for req in requests:
        try:
            generated_code, tokens_generated, finish_reason = await asyncio.to_thread(
                model_manager.generate,
                req.prompt,
                req.max_tokens,
                req.temperature,
                req.top_p,
                req.top_k,
                req.stop_sequences
            )
            results.append({
                "generated_code": generated_code,
                "tokens_generated": tokens_generated,
                "finish_reason": finish_reason,
                "error": None
            })
        except Exception as e:
            results.append({
                "generated_code": None,
                "tokens_generated": 0,
                "finish_reason": "error",
                "error": str(e)
            })

    return {"results": results}

@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "models": [
            {
                "id": "code-gen-v1",
                "name": "Code Generation Model v1",
                "version": "1.0.0",
                "loaded": model_manager.loaded,
                "capabilities": ["code-completion", "bash", "python"]
            }
        ]
    }

@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    return {
        "uptime_seconds": time.time() - model_manager.start_time,
        "model_loaded": model_manager.loaded,
        "device": Config.DEVICE,
        "total_requests": sum(len(v) for v in request_counts.values()),
        "active_rate_limits": len(request_counts)
    }

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")

    uvicorn.run(app, host=host, port=port)
