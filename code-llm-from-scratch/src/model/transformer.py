"""
PyTorch Transformer for Code Generation.

Key differences from MLX version:
- Uses torch.nn modules instead of mlx.nn
- MPS (Metal) backend for M1 Max acceleration
- More explicit dimension handling
- Compatible with HuggingFace ecosystem
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from .config import CoderConfig


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with causal masking.

    PyTorch implementation optimized for M1 Max MPS backend.
    """

    def __init__(self, config: CoderConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        # Combined QKV projection (more efficient)
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=config.use_bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)

        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Register causal mask buffer (not a parameter)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .view(1, 1, config.max_seq_len, config.max_seq_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # [batch, seq_len, 3 * d_model]

        # Split and reshape for multi-head
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, n_heads, seq_len, d_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [batch, n_heads, seq_len, seq_len]

        # Apply causal mask
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:, :, :seq_len, :seq_len] == 0,
            float('-inf')
        )

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        output = attn_weights @ v  # [batch, n_heads, seq_len, d_head]

        # Merge heads
        output = output.transpose(1, 2).contiguous()  # [batch, seq_len, n_heads, d_head]
        output = output.reshape(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.out_proj(output)
        output = self.resid_dropout(output)

        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, config: CoderConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=config.use_bias)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()  # GELU is standard for transformers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block with Pre-LayerNorm.

    Architecture:
        x = x + Attention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))
    """

    def __init__(self, config: CoderConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = MultiHeadAttention(config)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Attention block
        x = x + self.attn(self.ln1(x))

        # Feed-forward block
        x = x + self.ff(self.ln2(x))

        return x


class CodeTransformer(nn.Module):
    """
    Complete transformer model for code generation.

    PyTorch implementation with MPS backend support for M1 Max.
    """

    def __init__(self, config: CoderConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Positional embeddings (learned, like GPT-2)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # Dropout
        self.emb_dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # Output projection (language modeling head)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights (common practice)
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights (GPT-2 style)."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            targets: Target IDs for loss calculation (optional)

        Returns:
            Tuple of (logits, loss)
        """
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        token_emb = self.token_embedding(input_ids)  # [batch, seq_len, d_model]

        # Add positional embeddings
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        pos_emb = self.pos_embedding(positions)  # [seq_len, d_model]

        x = token_emb + pos_emb
        x = self.emb_dropout(x)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # Project to vocabulary
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]

        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1  # Ignore padding if used
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Starting tokens [batch_size, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            Generated token IDs [batch_size, seq_len + max_new_tokens]
        """
        if device is None:
            device = input_ids.device

        self.eval()

        for _ in range(max_new_tokens):
            # Crop if too long
            input_crop = input_ids
            if input_ids.shape[1] > self.config.max_seq_len:
                input_crop = input_ids[:, -self.config.max_seq_len:]

            # Forward pass
            logits, _ = self(input_crop)

            # Get logits for last token
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def count_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_device(self) -> torch.device:
        """Get device of model."""
        return next(self.parameters()).device


def create_model(
    model_size: str = 'tiny',
    vocab_size: int = 8000,
    device: Optional[str] = None
) -> CodeTransformer:
    """
    Create a code transformer model.

    Args:
        model_size: 'tiny', 'small', or 'medium'
        vocab_size: Vocabulary size
        device: Device to place model on ('mps', 'cuda', 'cpu', or None for auto)

    Returns:
        CodeTransformer model on specified device
    """
    from .config import get_model_config

    config = get_model_config(model_size, vocab_size)
    model = CodeTransformer(config)

    # Move to device
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    model = model.to(device)
    print(f"Model created: {model.count_parameters():,} parameters on {device}")

    return model
