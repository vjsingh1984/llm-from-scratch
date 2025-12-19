"""
Embedding layers for transformer models.

Includes token embeddings and various positional encoding strategies.
"""

import mlx.core as mx
import mlx.nn as nn
import math
from typing import Optional


class TokenEmbedding(nn.Module):
    """
    Token embedding layer.

    Converts discrete token IDs into continuous vector representations.
    Each token ID is mapped to a learnable d_model-dimensional vector.

    Args:
        vocab_size: Size of the vocabulary
        d_model: Dimension of the embedding vectors (model dimension)
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Learnable embedding matrix: [vocab_size, d_model]
        self.embedding = nn.Embedding(vocab_size, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Embed token IDs.

        Args:
            x: Token IDs of shape [batch_size, seq_len]

        Returns:
            Embedded tokens of shape [batch_size, seq_len, d_model]
        """
        # Scale embeddings by sqrt(d_model) as in "Attention Is All You Need"
        # This helps with gradient flow and matches the scale of positional encodings
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (from original Transformer paper).

    Adds position information to token embeddings using fixed sine/cosine functions.
    This allows the model to understand the order of tokens in the sequence.

    The encoding for position 'pos' and dimension 'i' is:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model: Dimension of the model
        max_seq_len: Maximum sequence length to pre-compute
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        # Pre-compute positional encodings for all positions
        # Shape: [max_seq_len, d_model]
        pe = mx.zeros((max_seq_len, d_model))

        # Create position indices: [0, 1, 2, ..., max_seq_len-1]
        position = mx.arange(0, max_seq_len).reshape(-1, 1)  # [max_seq_len, 1]

        # Create dimension indices and compute the div term
        # div_term = 1 / (10000^(2i/d_model)) for i in [0, 1, ..., d_model/2]
        div_term = mx.exp(
            mx.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )  # [d_model/2]

        # Apply sine to even indices
        pe[:, 0::2] = mx.sin(position * div_term)

        # Apply cosine to odd indices
        pe[:, 1::2] = mx.cos(position * div_term)

        # Store as buffer (not a trainable parameter)
        self.pe = pe

    def __call__(self, x: mx.array) -> mx.array:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input embeddings of shape [batch_size, seq_len, d_model]

        Returns:
            Embeddings with positional encoding, same shape as input
        """
        seq_len = x.shape[1]

        # Add positional encoding (broadcast across batch dimension)
        # self.pe[:seq_len] has shape [seq_len, d_model]
        # Broadcasting adds it to each batch element
        x = x + self.pe[:seq_len]

        return self.dropout(x)


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned positional embeddings (alternative to sinusoidal).

    Instead of fixed sine/cosine functions, this learns a unique embedding
    for each position. Used by models like GPT-2.

    Args:
        max_seq_len: Maximum sequence length
        d_model: Dimension of the model
    """

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # Learnable position embeddings: [max_seq_len, d_model]
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Add learned positional embeddings to input.

        Args:
            x: Input embeddings of shape [batch_size, seq_len, d_model]

        Returns:
            Embeddings with positional encoding, same shape as input
        """
        batch_size, seq_len, _ = x.shape

        # Create position indices: [0, 1, 2, ..., seq_len-1]
        positions = mx.arange(seq_len)

        # Get position embeddings: [seq_len, d_model]
        pos_emb = self.position_embedding(positions)

        # Add to input (broadcast across batch dimension)
        return x + pos_emb


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) - used in modern LLMs like LLaMA.

    RoPE applies a rotation to the query and key vectors based on their position.
    This provides relative position information and works well with extrapolation
    to longer sequences than seen during training.

    Args:
        d_head: Dimension of each attention head
        max_seq_len: Maximum sequence length
        base: Base for the geometric progression (default: 10000)
    """

    def __init__(self, d_head: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.base = base

        # Pre-compute inverse frequencies
        # Shape: [d_head // 2]
        inv_freq = 1.0 / (base ** (mx.arange(0, d_head, 2) / d_head))
        self.inv_freq = inv_freq

        # Pre-compute cos and sin for all positions
        self._precompute_freqs_cis()

    def _precompute_freqs_cis(self):
        """Pre-compute cos and sin values for all positions."""
        # Position indices: [0, 1, 2, ..., max_seq_len-1]
        t = mx.arange(self.max_seq_len)

        # Compute frequencies: [max_seq_len, d_head // 2]
        freqs = mx.outer(t, self.inv_freq)

        # Compute cos and sin
        self.cos_cached = mx.cos(freqs)  # [max_seq_len, d_head // 2]
        self.sin_cached = mx.sin(freqs)  # [max_seq_len, d_head // 2]

    def rotate_half(self, x: mx.array) -> mx.array:
        """
        Rotate half the hidden dimensions.

        Args:
            x: Input tensor [..., d_head]

        Returns:
            Rotated tensor of same shape
        """
        # Split into two halves
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]

        # Rotate: [-x2, x1]
        return mx.concatenate([-x2, x1], axis=-1)

    def __call__(self, q: mx.array, k: mx.array, seq_len: int) -> tuple[mx.array, mx.array]:
        """
        Apply rotary embeddings to queries and keys.

        Args:
            q: Query tensor of shape [batch_size, n_heads, seq_len, d_head]
            k: Key tensor of shape [batch_size, n_heads, seq_len, d_head]
            seq_len: Sequence length

        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        # Get cos and sin for current sequence length
        cos = self.cos_cached[:seq_len]  # [seq_len, d_head // 2]
        sin = self.sin_cached[:seq_len]  # [seq_len, d_head // 2]

        # Repeat for full d_head dimension
        cos = mx.repeat(cos, 2, axis=-1)  # [seq_len, d_head]
        sin = mx.repeat(sin, 2, axis=-1)  # [seq_len, d_head]

        # Reshape for broadcasting: [1, 1, seq_len, d_head]
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        # Apply rotation: x * cos + rotate_half(x) * sin
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)

        return q_rot, k_rot


# Helper function to create embeddings with positional encoding
def create_embeddings(
    vocab_size: int,
    d_model: int,
    max_seq_len: int,
    pos_encoding_type: str = "learned",
    dropout: float = 0.1
) -> tuple[TokenEmbedding, nn.Module]:
    """
    Create token embeddings and positional encoding.

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        max_seq_len: Maximum sequence length
        pos_encoding_type: Type of positional encoding
            - "sinusoidal": Fixed sine/cosine (original Transformer)
            - "learned": Learned embeddings (GPT-2 style)
            - "rope": Rotary Position Embedding (modern LLMs)
        dropout: Dropout probability

    Returns:
        Tuple of (token_embedding, positional_encoding)
    """
    token_emb = TokenEmbedding(vocab_size, d_model)

    if pos_encoding_type == "sinusoidal":
        pos_enc = PositionalEncoding(d_model, max_seq_len, dropout)
    elif pos_encoding_type == "learned":
        pos_enc = LearnedPositionalEmbedding(max_seq_len, d_model)
    elif pos_encoding_type == "rope":
        # RoPE is applied in attention layer, not here
        pos_enc = None
    else:
        raise ValueError(f"Unknown positional encoding type: {pos_encoding_type}")

    return token_emb, pos_enc
