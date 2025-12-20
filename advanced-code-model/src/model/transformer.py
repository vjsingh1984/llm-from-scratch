"""
PyTorch Transformer model for code generation optimized for Apple Silicon MPS.

Key optimizations for Apple Silicon:
1. MPS (Metal Performance Shaders) backend - native GPU acceleration
2. Efficient memory management
3. Mixed precision training support
4. Optimized for M1/M2/M3 chips
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from .config import ModelConfig
except ImportError:
    # When run as a script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.model.config import ModelConfig


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Faster and simpler alternative to LayerNorm used in LLaMA and other modern LLMs.
    Instead of normalizing by mean and variance, only normalizes by RMS.

    Benefits:
    - 10-15% faster than LayerNorm
    - Simpler computation (no mean subtraction)
    - Used in LLaMA, T5, and other state-of-the-art models

    Formula: RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS: sqrt(mean(x^2) + eps)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize and scale
        x_norm = x / rms
        return self.weight * x_norm


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) from "RoFormer: Enhanced Transformer with Rotary Position Embedding".

    Benefits over learned position embeddings:
    - Better length extrapolation (can handle sequences longer than training)
    - Relative position encoding (naturally captures position relationships)
    - No additional parameters
    - Used in LLaMA, GPT-NeoX, PaLM

    Paper: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, d_head: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute rotation matrix
        inv_freq = 1.0 / (self.base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin for all positions
        self._compute_cos_sin_cache(max_seq_len)

    def _compute_cos_sin_cache(self, seq_len: int):
        """Precompute cos and sin for all positions."""
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, d_head // 2)
        # Don't concatenate - we'll use this half-dimension directly
        # This matches the split q/k tensors in apply_rotary_pos_emb
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return cos and sin for rotary embeddings.

        Args:
            x: Input tensor (used only for device/dtype)
            seq_len: Sequence length

        Returns:
            Tuple of (cos, sin) tensors of shape (seq_len, d_head // 2)
        """
        # Extend cache if needed
        if seq_len > self.cos_cached.shape[0]:
            self._compute_cos_sin_cache(seq_len)

        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len]
        )


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor [batch, n_heads, seq_len, d_head]
        k: Key tensor [batch, n_heads, seq_len, d_head]
        cos: Cosine values [seq_len, d_head // 2]
        sin: Sine values [seq_len, d_head // 2]

    Returns:
        Tuple of rotated (q, k) tensors
    """
    # Reshape cos/sin to match split q/k dimensions
    # [seq_len, d_head // 2] -> [1, 1, seq_len, d_head // 2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Split into first and second half
    q_half1, q_half2 = q.chunk(2, dim=-1)
    k_half1, k_half2 = k.chunk(2, dim=-1)

    # Apply rotation
    q_rotated = torch.cat([
        q_half1 * cos - q_half2 * sin,
        q_half1 * sin + q_half2 * cos
    ], dim=-1)

    k_rotated = torch.cat([
        k_half1 * cos - k_half2 * sin,
        k_half1 * sin + k_half2 * cos
    ], dim=-1)

    return q_rotated, k_rotated


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with causal masking.

    Optimized for Apple Silicon MPS backend.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_head

        assert self.d_model % self.n_heads == 0

        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model, bias=True)

        # Output projection
        self.out_proj = nn.Linear(self.d_model, self.d_model)

        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)

        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.d_head)

        # Rotary position embeddings (if enabled)
        self.rope = None
        if hasattr(config, 'use_rope') and config.use_rope:
            self.rope = RotaryPositionEmbedding(self.d_head, config.max_seq_len)

        # Register causal mask buffer
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1).bool()
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # [batch, seq_len, 3 * d_model]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, n_heads, seq_len, d_head]

        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE if enabled
        if self.rope is not None:
            cos, sin = self.rope(q, seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention
        # scores = (Q @ K.T) / sqrt(d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch, n_heads, seq_len, seq_len]

        # Apply causal mask
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(causal_mask, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # [batch, n_heads, seq_len, d_head]

        # Reshape and project output
        out = out.transpose(1, 2)  # [batch, seq_len, n_heads, d_head]
        out = out.reshape(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Uses GELU activation and two linear transformations.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-normalization.

    Architecture:
        x = x + Attention(Norm(x))
        x = x + FFN(Norm(x))

    Supports both LayerNorm and RMSNorm (faster alternative).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Pre-normalization (LayerNorm or RMSNorm)
        norm_class = RMSNorm if config.use_rmsnorm else nn.LayerNorm
        self.ln1 = norm_class(config.d_model)
        self.ln2 = norm_class(config.d_model)

        # Main components
        self.attention = MultiHeadAttention(config)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Use gradient checkpointing if enabled (saves memory at cost of recomputation)
        if self.config.use_gradient_checkpointing and self.training:
            # Attention block with checkpointing
            x = x + checkpoint(
                lambda x_in: self.attention(self.ln1(x_in), mask=mask),
                x,
                use_reentrant=False
            )

            # Feed-forward block with checkpointing
            x = x + checkpoint(
                lambda x_in: self.ffn(self.ln2(x_in)),
                x,
                use_reentrant=False
            )
        else:
            # Standard forward pass (no checkpointing)
            # Attention block with residual connection
            x = x + self.attention(self.ln1(x), mask=mask)

            # Feed-forward block with residual connection
            x = x + self.ffn(self.ln2(x))

        return x


class Transformer(nn.Module):
    """
    Complete transformer model for code generation.

    Optimized for PyTorch with MPS backend on Apple Silicon.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # Embedding dropout
        self.emb_dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final normalization (LayerNorm or RMSNorm)
        norm_class = RMSNorm if config.use_rmsnorm else nn.LayerNorm
        self.ln_f = norm_class(config.d_model)

        # Language modeling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie embeddings (share weights between input and output embeddings)
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Print model info
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Initialized PyTorch Transformer:")
        print(f"  Parameters: {num_params / 1e6:.1f}M")
        print(f"  Layers: {config.n_layers}")
        print(f"  Attention heads: {config.n_heads}")

    def _init_weights(self, module):
        """Initialize weights with normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token indices [batch, seq_len]
            mask: Optional attention mask

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Get embeddings
        token_emb = self.token_embedding(input_ids)

        # Position embeddings
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device)
        pos_emb = self.position_embedding(positions)

        # Combine embeddings
        x = token_emb + pos_emb
        x = self.emb_dropout(x)

        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask)

        # Final layer norm
        x = self.ln_f(x)

        # Project to vocabulary
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_length: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """
        Generate code autoregressively.

        Args:
            prompt_ids: Initial tokens [batch, prompt_len]
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens
            top_p: Nucleus sampling threshold

        Returns:
            Generated token ids [batch, prompt_len + max_length]
        """
        self.eval()
        batch_size = prompt_ids.shape[0]
        device = prompt_ids.device
        generated = prompt_ids

        for _ in range(max_length):
            # Truncate if necessary
            if generated.shape[1] > self.config.max_seq_len:
                generated = generated[:, -self.config.max_seq_len:]

            # Get logits for next token
            logits = self(generated)
            next_token_logits = logits[:, -1, :] / temperature  # [batch, vocab_size]

            # Top-k filtering
            if top_k > 0:
                top_k_vals, _ = torch.topk(next_token_logits, top_k, dim=-1)
                threshold = top_k_vals[:, -1:]
                next_token_logits = torch.where(
                    next_token_logits < threshold,
                    torch.full_like(next_token_logits, float('-inf')),
                    next_token_logits
                )

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False

                # Scatter back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))

            # Sample from distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def num_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config: ModelConfig, device: str = 'mps') -> Transformer:
    """
    Create and initialize PyTorch transformer model.

    Args:
        config: Model configuration
        device: Device to place model on ('mps', 'cuda', or 'cpu')

    Returns:
        Initialized Transformer on specified device
    """
    model = Transformer(config)

    # Move to device
    if device == 'mps' and torch.backends.mps.is_available():
        model = model.to('mps')
        print(f"  Device: MPS (Apple Silicon GPU)")
    elif device == 'cuda' and torch.cuda.is_available():
        model = model.to('cuda')
        print(f"  Device: CUDA GPU")
    else:
        model = model.to('cpu')
        print(f"  Device: CPU")

    return model


if __name__ == '__main__':
    """Test model creation and forward pass."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.model.config import get_medium_config

    print("Testing PyTorch Transformer Model")
    print("=" * 60)

    # Create config
    config = get_medium_config()

    # Create model
    model = create_model(config, device='mps')

    print()
    print("Model created successfully!")
    print(f"Total parameters: {model.num_parameters() / 1e6:.1f}M")

    # Test forward pass
    batch_size = 2
    seq_len = 128
    device = next(model.parameters()).device
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    print()
    print(f"Testing forward pass...")
    print(f"  Input shape: {input_ids.shape}")

    logits = model(input_ids)
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected: ({batch_size}, {seq_len}, {config.vocab_size})")

    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    print()
    print("✓ Forward pass successful!")

    # Test generation
    print()
    print("Testing generation...")
    prompt = input_ids[:, :10]  # Use first 10 tokens as prompt
    generated = model.generate(prompt, max_length=20)
    print(f"  Prompt shape: {prompt.shape}")
    print(f"  Generated shape: {generated.shape}")
    print("✓ Generation successful!")
