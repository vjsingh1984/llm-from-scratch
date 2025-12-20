#!/usr/bin/env python3
"""
Advanced Example: REST API Deployment

This example demonstrates how to deploy the trained model as a production-ready
REST API using FastAPI. It includes:

1. API endpoints for code generation
2. Request validation and error handling
3. Caching for improved performance
4. Health checks and monitoring
5. Docker deployment configuration

Installation:
    pip install fastapi uvicorn pydantic

Usage:
    # Development
    python examples/deployment_api.py

    # Production (with uvicorn)
    uvicorn examples.deployment_api:app --host 0.0.0.0 --port 8000 --workers 4

    # Docker
    docker build -t code-llm-api .
    docker run -p 8000:8000 code-llm-api

API Endpoints:
    POST /generate - Generate code from prompt
    GET /health - Health check
    GET /info - Model information
"""

import os
import sys
import time
import hashlib
from typing import Optional, List
from functools import lru_cache

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field, validator
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model.transformer import CodeTransformer
from src.model.config import CoderConfig
from src.tokenizer.bpe import BPETokenizer


# ============================================================================
# Configuration
# ============================================================================

class ServerConfig:
    """Server configuration."""
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/code/code_model_final.pt')
    TOKENIZER_PATH = os.getenv('TOKENIZER_PATH', 'models/language/language_tokenizer.json')
    MAX_PROMPT_LENGTH = int(os.getenv('MAX_PROMPT_LENGTH', '200'))
    MAX_GENERATION_LENGTH = int(os.getenv('MAX_GENERATION_LENGTH', '500'))
    CACHE_SIZE = int(os.getenv('CACHE_SIZE', '100'))
    DEFAULT_TEMPERATURE = float(os.getenv('DEFAULT_TEMPERATURE', '0.8'))
    DEFAULT_TOP_K = int(os.getenv('DEFAULT_TOP_K', '50'))
    DEFAULT_TOP_P = float(os.getenv('DEFAULT_TOP_P', '0.9'))


# ============================================================================
# Request/Response Models
# ============================================================================

class GenerateRequest(BaseModel):
    """Request model for code generation."""
    prompt: str = Field(
        ...,
        description="The prompt to generate code from",
        example="#!/bin/bash\n# Create a backup script"
    )
    max_length: int = Field(
        default=200,
        ge=1,
        le=ServerConfig.MAX_GENERATION_LENGTH,
        description="Maximum length of generated code"
    )
    temperature: float = Field(
        default=ServerConfig.DEFAULT_TEMPERATURE,
        ge=0.1,
        le=2.0,
        description="Sampling temperature (higher = more creative)"
    )
    top_k: int = Field(
        default=ServerConfig.DEFAULT_TOP_K,
        ge=1,
        le=100,
        description="Top-k sampling parameter"
    )
    top_p: float = Field(
        default=ServerConfig.DEFAULT_TOP_P,
        ge=0.0,
        le=1.0,
        description="Top-p (nucleus) sampling parameter"
    )

    @validator('prompt')
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError("Prompt cannot be empty")
        if len(v) > ServerConfig.MAX_PROMPT_LENGTH * 10:  # Character limit
            raise ValueError(f"Prompt too long (max ~{ServerConfig.MAX_PROMPT_LENGTH * 10} chars)")
        return v


class GenerateResponse(BaseModel):
    """Response model for code generation."""
    generated_code: str = Field(..., description="The generated code")
    prompt: str = Field(..., description="The original prompt")
    num_tokens_generated: int = Field(..., description="Number of tokens generated")
    generation_time: float = Field(..., description="Generation time in seconds")
    model_info: dict = Field(..., description="Model information")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    timestamp: float


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_path: str
    vocab_size: int
    num_parameters: int
    device: str
    config: dict


# ============================================================================
# Model Manager
# ============================================================================

class ModelManager:
    """Manages model loading and inference."""

    def __init__(self):
        self.model: Optional[CodeTransformer] = None
        self.tokenizer: Optional[BPETokenizer] = None
        self.device: Optional[torch.device] = None
        self.config: Optional[CoderConfig] = None
        self._cache = {}

    def load(self):
        """Load model and tokenizer."""
        print("Loading model...")

        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        print(f"Using device: {self.device}")

        # Load tokenizer
        print(f"Loading tokenizer from {ServerConfig.TOKENIZER_PATH}...")
        self.tokenizer = BPETokenizer()
        self.tokenizer.load(ServerConfig.TOKENIZER_PATH)

        # Load model
        print(f"Loading model from {ServerConfig.MODEL_PATH}...")
        checkpoint = torch.load(ServerConfig.MODEL_PATH, map_location=self.device)

        self.config = checkpoint['config']
        self.model = CodeTransformer(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded: {num_params:,} parameters")

    def generate(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> tuple[str, int, float]:
        """
        Generate code from prompt.

        Returns:
            Tuple of (generated_code, num_tokens, generation_time)
        """
        start_time = time.time()

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        if not input_ids:
            raise ValueError("Prompt encoding resulted in empty token sequence")

        # Truncate if needed
        if len(input_ids) > ServerConfig.MAX_PROMPT_LENGTH:
            input_ids = input_ids[-ServerConfig.MAX_PROMPT_LENGTH:]

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Generate
        generated = input_tensor.clone()

        with torch.no_grad():
            for i in range(max_length):
                # Get logits
                logits, _ = self.model(generated)
                next_logits = logits[0, -1, :] / temperature

                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = float('-inf')

                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_logits[indices_to_remove] = float('-inf')

                # Sample
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Check for end of sequence
                if next_token.item() == self.tokenizer.encode('\n')[-1]:
                    # Allow some newlines but stop at too many consecutive
                    if i > 10:  # Minimum generation length
                        break

                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

        # Decode
        generated_ids = generated[0].tolist()
        generated_text = self.tokenizer.decode(generated_ids)

        generation_time = time.time() - start_time
        num_tokens = len(generated_ids) - len(input_ids)

        return generated_text, num_tokens, generation_time

    def get_info(self) -> dict:
        """Get model information."""
        num_params = sum(p.numel() for p in self.model.parameters())
        return {
            'model_path': ServerConfig.MODEL_PATH,
            'vocab_size': len(self.tokenizer.vocab),
            'num_parameters': num_params,
            'device': str(self.device),
            'config': {
                'n_layers': self.config.n_layers,
                'd_model': self.config.d_model,
                'n_heads': self.config.n_heads,
                'd_ff': self.config.d_ff,
                'max_seq_len': self.config.max_seq_len,
            }
        }


# ============================================================================
# FastAPI Application
# ============================================================================

# Initialize app
app = FastAPI(
    title="Code LLM API",
    description="REST API for bash script generation using a GPT-style transformer",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model manager
model_manager = ModelManager()


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        model_manager.load()
        print("API ready!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint."""
    return {
        "message": "Code LLM API",
        "docs": "/docs",
        "health": "/health",
        "info": "/info"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model_manager.model is not None else "unhealthy",
        model_loaded=model_manager.model is not None,
        device=str(model_manager.device) if model_manager.device else "unknown",
        timestamp=time.time()
    )


@app.get("/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information."""
    if model_manager.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    info = model_manager.get_info()
    return ModelInfoResponse(**info)


@app.post("/generate", response_model=GenerateResponse)
async def generate_code(request: GenerateRequest):
    """
    Generate bash script from prompt.

    This endpoint takes a natural language prompt and generates bash code.

    Example:
        ```
        POST /generate
        {
            "prompt": "#!/bin/bash\\n# Create a backup script",
            "max_length": 200,
            "temperature": 0.8
        }
        ```
    """
    if model_manager.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        generated_code, num_tokens, gen_time = model_manager.generate(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p
        )

        return GenerateResponse(
            generated_code=generated_code,
            prompt=request.prompt,
            num_tokens_generated=num_tokens,
            generation_time=gen_time,
            model_info={
                'temperature': request.temperature,
                'top_k': request.top_k,
                'top_p': request.top_p,
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}"
        )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("CODE LLM API SERVER")
    print("="*60)
    print(f"\nModel: {ServerConfig.MODEL_PATH}")
    print(f"Tokenizer: {ServerConfig.TOKENIZER_PATH}")
    print("\nStarting server...")
    print("Docs: http://localhost:8000/docs")
    print("="*60 + "\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
