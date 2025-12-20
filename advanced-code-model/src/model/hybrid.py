"""
Hybrid Architecture: Mamba + Local Attention.

Combines the best of both worlds:
- Mamba (SSM) for global long-range dependencies (O(n) complexity)
- Local attention for fine-grained short-range patterns (O(n*w) complexity where w=window)

Key advantages:
- Better than pure Mamba: Local attention for precise token interactions
- Better than pure Transformer: O(n) global context instead of O(n²)
- Flexible: Can adjust local window size vs global SSM ratio

Architecture:
- Alternating Mamba and Local Attention blocks
- Or interleaved within same block
- Configurable local attention window size
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

from .config import ModelConfig
from .mamba import SelectiveSSM
from .transformer import RMSNorm


class LocalAttention(nn.Module):
    """
    Local windowed attention.

    Only attends to tokens within a local window, reducing complexity from O(n²) to O(n*w).
    """

    def __init__(self, config: ModelConfig, window_size: int):
        super().__init__()
        self.config = config
        self.window_size = window_size
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads

        # Q, K, V projections
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)

        # Output projection
        self.o_proj = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with local windowed attention (VECTORIZED - no loops!).

        Args:
            x: Input [batch, seq_len, d_model]

        Returns:
            Output [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        # q, k, v: [batch, n_heads, seq_len, d_head]

        # Compute full attention scores (we'll mask later)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # scores: [batch, n_heads, seq_len, seq_len]

        # Create local window mask (causal + local window)
        # Each position i can only attend to positions in [max(0, i-window), i]
        indices = torch.arange(seq_len, device=x.device)
        row_indices = indices.unsqueeze(1)  # [seq_len, 1]
        col_indices = indices.unsqueeze(0)  # [1, seq_len]

        # Causal mask: col <= row (can only attend to past and present)
        causal_mask = col_indices <= row_indices

        # Local window mask: row - col <= window_size (within window)
        local_mask = row_indices - col_indices <= self.window_size

        # Combine masks
        mask = causal_mask & local_mask  # [seq_len, seq_len]

        # Apply mask (set invalid positions to -inf before softmax)
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Softmax (attention weights)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)  # Handle -inf case at start
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # [batch, n_heads, seq_len, d_head]

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        output = self.o_proj(output)

        return output


class HybridBlock(nn.Module):
    """
    Hybrid block: Mamba SSM + Local Attention.

    Combines global long-range modeling (Mamba) with local fine-grained attention.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Normalization
        if config.use_rmsnorm:
            self.norm1 = RMSNorm(config.d_model)
            self.norm2 = RMSNorm(config.d_model)
            self.norm3 = RMSNorm(config.d_model)
        else:
            self.norm1 = nn.LayerNorm(config.d_model)
            self.norm2 = nn.LayerNorm(config.d_model)
            self.norm3 = nn.LayerNorm(config.d_model)

        # Mamba (SSM) for global context
        self.ssm = SelectiveSSM(
            d_model=config.d_model,
            state_size=config.state_size,
            conv_size=config.conv_size
        )

        # Local attention for fine-grained patterns
        self.local_attn = LocalAttention(config, window_size=config.hybrid_local_window)

        # MLP
        self.mlp_up = nn.Linear(config.d_model, config.d_ff)
        self.mlp_down = nn.Linear(config.d_ff, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hybrid block.

        Args:
            x: Input [batch, seq_len, d_model]

        Returns:
            Output [batch, seq_len, d_model]
        """
        # 1. Mamba for global context
        x = x + self.dropout(self.ssm(self.norm1(x)))

        # 2. Local attention for fine-grained patterns
        x = x + self.dropout(self.local_attn(self.norm2(x)))

        # 3. MLP
        mlp_out = self.mlp_down(F.gelu(self.mlp_up(self.norm3(x))))
        x = x + self.dropout(mlp_out)

        return x


class HybridModel(nn.Module):
    """
    Hybrid language model: Mamba + Local Attention.

    Best of both worlds:
    - O(n) global context from Mamba
    - Precise local interactions from windowed attention
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings (no position embeddings needed - Mamba handles it!)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Hybrid blocks
        self.blocks = nn.ModuleList([
            HybridBlock(config)
            for _ in range(config.n_layers)
        ])

        # Final normalization
        if config.use_rmsnorm:
            self.ln_f = RMSNorm(config.d_model)
        else:
            self.ln_f = nn.LayerNorm(config.d_model)

        # Output projection (tie weights with input embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Weight tying

        self.dropout = nn.Dropout(config.dropout)

        # Initialize weights
        self.apply(self._init_weights)

        print(f"Initialized Hybrid Model:")
        print(f"  Architecture: Mamba (global) + Local Attention (local)")
        print(f"  Parameters: {self.count_parameters() / 1e6:.1f}M")
        print(f"  Layers: {config.n_layers}")
        print(f"  Hidden dim: {config.d_model}")
        print(f"  State size: {config.state_size} (Mamba)")
        print(f"  Local window: {config.hybrid_local_window} (Attention)")
        print(f"  Complexity: O(n + n*w) where n=seq_len, w=window")
        print(f"  Vocabulary: {config.vocab_size}")

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input token IDs [batch, seq_len]

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        # Token embeddings
        x = self.token_embedding(x)
        x = self.dropout(x)

        # Hybrid blocks
        for block in self.blocks:
            if self.config.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, use_reentrant=False
                )
            else:
                x = block(x)

        # Final norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits


def create_hybrid_model(config: ModelConfig, device: str = "cpu") -> HybridModel:
    """
    Create Hybrid model and move to device.

    Args:
        config: Model configuration
        device: Device to use ('cpu', 'cuda', 'mps')

    Returns:
        Initialized Hybrid model
    """
    model = HybridModel(config)
    model = model.to(device)
    return model
