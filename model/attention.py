"""
Multi-Head Self-Attention implementation.

The core mechanism that allows transformers to model relationships between tokens.
"""

import mlx.core as mx
import mlx.nn as nn
import math
from typing import Optional


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.

    Key concepts:
    1. Self-Attention: Each token attends to all other tokens (including itself)
    2. Multi-Head: Run multiple attention mechanisms in parallel
    3. Causal Masking: Prevent tokens from attending to future tokens (for autoregressive models)

    The attention mechanism:
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Where:
        - Q (Query): "What am I looking for?"
        - K (Key): "What do I contain?"
        - V (Value): "What do I actually represent?"

    Args:
        d_model: Model dimension (input/output dimension)
        n_heads: Number of attention heads
        dropout: Dropout probability (default: 0.1)
        use_bias: Whether to use bias in linear projections (default: True)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_bias: bool = True
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # Dimension per head

        # Single linear layer for Q, K, V projections (more efficient)
        # Output dimension is 3 * d_model (for Q, K, V)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=use_bias)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=use_bias)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Scale factor for attention scores
        self.scale = 1.0 / math.sqrt(self.d_head)

    def create_causal_mask(self, seq_len: int) -> mx.array:
        """
        Create causal attention mask to prevent attending to future tokens.

        The mask is a lower triangular matrix of ones, with upper triangle as zeros.
        When applied, future positions are masked out with -inf before softmax.

        Args:
            seq_len: Sequence length

        Returns:
            Causal mask of shape [seq_len, seq_len]
        """
        # Create lower triangular matrix
        # mask[i, j] = 1 if j <= i, else 0
        mask = mx.tril(mx.ones((seq_len, seq_len)))
        return mask

    def split_heads(self, x: mx.array) -> mx.array:
        """
        Split the last dimension into (n_heads, d_head).

        Args:
            x: Input of shape [batch_size, seq_len, d_model]

        Returns:
            Output of shape [batch_size, n_heads, seq_len, d_head]
        """
        batch_size, seq_len, _ = x.shape

        # Reshape to [batch_size, seq_len, n_heads, d_head]
        x = x.reshape(batch_size, seq_len, self.n_heads, self.d_head)

        # Transpose to [batch_size, n_heads, seq_len, d_head]
        x = x.transpose(0, 2, 1, 3)

        return x

    def merge_heads(self, x: mx.array) -> mx.array:
        """
        Merge the head dimension back.

        Args:
            x: Input of shape [batch_size, n_heads, seq_len, d_head]

        Returns:
            Output of shape [batch_size, seq_len, d_model]
        """
        batch_size, _, seq_len, _ = x.shape

        # Transpose to [batch_size, seq_len, n_heads, d_head]
        x = x.transpose(0, 2, 1, 3)

        # Reshape to [batch_size, seq_len, d_model]
        x = x.reshape(batch_size, seq_len, self.d_model)

        return x

    def attention(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        mask: Optional[mx.array] = None
    ) -> mx.array:
        """
        Scaled dot-product attention.

        Args:
            q: Queries of shape [batch_size, n_heads, seq_len, d_head]
            k: Keys of shape [batch_size, n_heads, seq_len, d_head]
            v: Values of shape [batch_size, n_heads, seq_len, d_head]
            mask: Optional attention mask

        Returns:
            Attention output of shape [batch_size, n_heads, seq_len, d_head]
        """
        # Compute attention scores: Q @ K^T
        # Shape: [batch_size, n_heads, seq_len, seq_len]
        attn_scores = q @ k.transpose(0, 1, 3, 2)

        # Scale by sqrt(d_head)
        attn_scores = attn_scores * self.scale

        # Apply causal mask if provided
        if mask is not None:
            # Where mask is 0, set attention score to -inf
            # This makes softmax output 0 for those positions
            attn_scores = mx.where(mask, attn_scores, -1e9)

        # Apply softmax to get attention weights
        # Shape: [batch_size, n_heads, seq_len, seq_len]
        attn_weights = mx.softmax(attn_scores, axis=-1)

        # Apply dropout to attention weights
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention weights to values
        # Shape: [batch_size, n_heads, seq_len, d_head]
        output = attn_weights @ v

        return output

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        use_causal_mask: bool = True
    ) -> mx.array:
        """
        Forward pass of multi-head attention.

        Args:
            x: Input of shape [batch_size, seq_len, d_model]
            mask: Optional attention mask of shape [seq_len, seq_len]
            use_causal_mask: Whether to apply causal masking (default: True)

        Returns:
            Output of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        # Shape: [batch_size, seq_len, 3 * d_model]
        qkv = self.qkv_proj(x)

        # Split into Q, K, V
        # Each has shape: [batch_size, seq_len, d_model]
        q, k, v = mx.split(qkv, 3, axis=-1)

        # Split heads
        # Shape: [batch_size, n_heads, seq_len, d_head]
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Create causal mask if needed
        if use_causal_mask and mask is None:
            mask = self.create_causal_mask(seq_len)

        # Apply attention
        # Shape: [batch_size, n_heads, seq_len, d_head]
        attn_output = self.attention(q, k, v, mask)

        # Merge heads
        # Shape: [batch_size, seq_len, d_model]
        output = self.merge_heads(attn_output)

        # Output projection
        output = self.out_proj(output)

        # Residual dropout
        output = self.resid_dropout(output)

        return output


class FlashAttention(nn.Module):
    """
    Flash Attention - memory-efficient attention mechanism.

    Flash Attention reduces memory usage and improves speed by:
    1. Fusing operations to reduce memory reads/writes
    2. Tiling the computation to fit in fast SRAM
    3. Recomputing attention scores during backward pass instead of storing them

    This is a simplified version. For production, use optimized implementations.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        # For now, we'll use the standard attention implementation
        # MLX may provide optimized attention in future versions
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        use_causal_mask: bool = True
    ) -> mx.array:
        """Forward pass."""
        return self.attn(x, mask, use_causal_mask)


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) - used in modern LLMs like LLaMA 2.

    Instead of having separate K and V for each head, GQA shares K and V
    across groups of query heads. This reduces memory and compute.

    - Multi-Head Attention: n_kv_heads = n_heads (one K, V per head)
    - Grouped Query Attention: n_kv_heads < n_heads (K, V shared across groups)
    - Multi-Query Attention: n_kv_heads = 1 (single K, V for all heads)

    Args:
        d_model: Model dimension
        n_heads: Number of query heads
        n_kv_heads: Number of key-value heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int = None,
        dropout: float = 0.1,
        use_bias: bool = True
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Default to multi-head attention if n_kv_heads not specified
        if n_kv_heads is None:
            n_kv_heads = n_heads

        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads  # Number of query heads per KV head
        self.d_head = d_model // n_heads

        # Separate projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=use_bias)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=use_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=use_bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_head)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        use_causal_mask: bool = True
    ) -> mx.array:
        """Forward pass with grouped query attention."""
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x)  # [batch, seq_len, d_model]
        k = self.k_proj(x)  # [batch, seq_len, n_kv_heads * d_head]
        v = self.v_proj(x)  # [batch, seq_len, n_kv_heads * d_head]

        # Reshape Q: [batch, n_heads, seq_len, d_head]
        q = q.reshape(batch_size, seq_len, self.n_heads, self.d_head)
        q = q.transpose(0, 2, 1, 3)

        # Reshape K, V: [batch, n_kv_heads, seq_len, d_head]
        k = k.reshape(batch_size, seq_len, self.n_kv_heads, self.d_head)
        k = k.transpose(0, 2, 1, 3)

        v = v.reshape(batch_size, seq_len, self.n_kv_heads, self.d_head)
        v = v.transpose(0, 2, 1, 3)

        # Repeat K and V for each group
        # [batch, n_kv_heads, seq_len, d_head] -> [batch, n_heads, seq_len, d_head]
        if self.n_groups > 1:
            k = mx.repeat(k, self.n_groups, axis=1)
            v = mx.repeat(v, self.n_groups, axis=1)

        # Compute attention
        attn_scores = q @ k.transpose(0, 1, 3, 2) * self.scale

        if use_causal_mask:
            causal_mask = mx.tril(mx.ones((seq_len, seq_len)))
            attn_scores = mx.where(causal_mask, attn_scores, -1e9)

        attn_weights = mx.softmax(attn_scores, axis=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = attn_weights @ v  # [batch, n_heads, seq_len, d_head]

        # Merge heads
        output = output.transpose(0, 2, 1, 3)
        output = output.reshape(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.out_proj(output)
        output = self.resid_dropout(output)

        return output
