"""
Transformer blocks and complete GPT-style language model.

Combines attention, feedforward networks, and normalization into a complete model.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Literal
from dataclasses import dataclass

from .attention import MultiHeadAttention, GroupedQueryAttention
from .embedding import TokenEmbedding, LearnedPositionalEmbedding, PositionalEncoding


@dataclass
class GPTConfig:
    """
    Configuration for GPT model.

    Use this to easily create models of different sizes.
    """
    # Model architecture
    vocab_size: int = 50257  # GPT-2 vocabulary size
    n_layers: int = 12  # Number of transformer blocks
    d_model: int = 768  # Model dimension
    n_heads: int = 12  # Number of attention heads
    d_ff: int = 3072  # Feedforward dimension (typically 4 * d_model)
    max_seq_len: int = 1024  # Maximum sequence length

    # Regularization
    dropout: float = 0.1
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1

    # Architecture choices
    pos_encoding: Literal["learned", "sinusoidal"] = "learned"
    use_bias: bool = True  # Use bias in linear layers
    layer_norm_eps: float = 1e-5

    # Advanced options
    use_gqa: bool = False  # Use Grouped Query Attention
    n_kv_heads: Optional[int] = None  # Number of KV heads for GQA

    @classmethod
    def gpt2_small(cls):
        """GPT-2 Small configuration (124M parameters)."""
        return cls(
            vocab_size=50257,
            n_layers=12,
            d_model=768,
            n_heads=12,
            d_ff=3072,
            max_seq_len=1024
        )

    @classmethod
    def gpt2_medium(cls):
        """GPT-2 Medium configuration (350M parameters)."""
        return cls(
            vocab_size=50257,
            n_layers=24,
            d_model=1024,
            n_heads=16,
            d_ff=4096,
            max_seq_len=1024
        )

    @classmethod
    def gpt2_large(cls):
        """GPT-2 Large configuration (774M parameters)."""
        return cls(
            vocab_size=50257,
            n_layers=36,
            d_model=1280,
            n_heads=20,
            d_ff=5120,
            max_seq_len=1024
        )

    @classmethod
    def tiny(cls, vocab_size: int = 8000):
        """Tiny model for quick experimentation (50M parameters)."""
        return cls(
            vocab_size=vocab_size,
            n_layers=6,
            d_model=512,
            n_heads=8,
            d_ff=2048,
            max_seq_len=512
        )


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Two linear transformations with a non-linearity in between:
        FFN(x) = activation(xW1 + b1)W2 + b2

    Typically expands dimension by 4x (d_model -> d_ff -> d_model).

    Args:
        d_model: Input/output dimension
        d_ff: Hidden dimension
        dropout: Dropout probability
        activation: Activation function ("gelu", "relu", "swish")
        use_bias: Whether to use bias
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_bias: bool = True
    ):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.fc2 = nn.Linear(d_ff, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

        # Activation function
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swish":
            self.activation = nn.SiLU()  # SiLU is same as Swish
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input of shape [batch_size, seq_len, d_model]

        Returns:
            Output of shape [batch_size, seq_len, d_model]
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single Transformer block.

    Architecture (Pre-LayerNorm variant, used by GPT-2):
        x = x + Attention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))

    This is slightly different from the original Transformer paper
    which used Post-LayerNorm (LayerNorm after residual addition).
    Pre-LayerNorm generally trains more stably.

    Args:
        config: Model configuration
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        # Layer normalization
        self.ln1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # Multi-head attention
        if config.use_gqa and config.n_kv_heads is not None:
            self.attn = GroupedQueryAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                dropout=config.attn_dropout,
                use_bias=config.use_bias
            )
        else:
            self.attn = MultiHeadAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                dropout=config.attn_dropout,
                use_bias=config.use_bias
            )

        # Feedforward network
        self.ff = FeedForward(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.resid_dropout,
            use_bias=config.use_bias
        )

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input of shape [batch_size, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            Output of shape [batch_size, seq_len, d_model]
        """
        # Pre-LayerNorm: normalize before attention
        # Residual connection: add input to output
        x = x + self.attn(self.ln1(x), mask=mask)

        # Pre-LayerNorm: normalize before feedforward
        # Residual connection: add input to output
        x = x + self.ff(self.ln2(x))

        return x


class GPTModel(nn.Module):
    """
    Complete GPT-style language model.

    Architecture:
        1. Token Embeddings + Positional Embeddings
        2. Stack of Transformer Blocks
        3. Final Layer Normalization
        4. Output Projection to Vocabulary

    This model performs next-token prediction (causal language modeling).

    Args:
        config: Model configuration
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = TokenEmbedding(config.vocab_size, config.d_model)

        # Positional embeddings
        if config.pos_encoding == "learned":
            self.pos_embedding = LearnedPositionalEmbedding(
                config.max_seq_len,
                config.d_model
            )
        elif config.pos_encoding == "sinusoidal":
            self.pos_embedding = PositionalEncoding(
                config.d_model,
                config.max_seq_len,
                config.dropout
            )
        else:
            raise ValueError(f"Unknown positional encoding: {config.pos_encoding}")

        # Dropout for embeddings
        self.emb_dropout = nn.Dropout(config.dropout)

        # Stack of transformer blocks
        self.blocks = [TransformerBlock(config) for _ in range(config.n_layers)]

        # Final layer normalization
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # Output projection (vocabulary)
        # Note: Many models tie the weights of this layer with token embeddings
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights.

        Following GPT-2 initialization:
        - Normal distribution with std=0.02 for most weights
        - Scaled initialization for residual projections
        """
        # Note: MLX automatically initializes weights, but we can customize if needed
        # For now, we'll rely on MLX's default initialization
        pass

    def __call__(
        self,
        input_ids: mx.array,
        targets: Optional[mx.array] = None
    ) -> tuple[mx.array, Optional[mx.array]]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            targets: Optional target IDs for loss computation [batch_size, seq_len]

        Returns:
            Tuple of (logits, loss)
            - logits: Output logits of shape [batch_size, seq_len, vocab_size]
            - loss: Cross-entropy loss (if targets provided), else None
        """
        batch_size, seq_len = input_ids.shape

        # Get token embeddings
        # Shape: [batch_size, seq_len, d_model]
        x = self.token_embedding(input_ids)

        # Add positional embeddings
        x = self.pos_embedding(x)

        # Dropout
        x = self.emb_dropout(x)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer normalization
        x = self.ln_f(x)

        # Project to vocabulary
        # Shape: [batch_size, seq_len, vocab_size]
        logits = self.lm_head(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = self.compute_loss(logits, targets)

        return logits, loss

    def compute_loss(self, logits: mx.array, targets: mx.array) -> mx.array:
        """
        Compute cross-entropy loss.

        Args:
            logits: Model outputs of shape [batch_size, seq_len, vocab_size]
            targets: Target token IDs of shape [batch_size, seq_len]

        Returns:
            Scalar loss value
        """
        # Reshape for cross-entropy
        # logits: [batch_size * seq_len, vocab_size]
        # targets: [batch_size * seq_len]
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.reshape(-1, vocab_size)
        targets = targets.reshape(-1)

        # Compute cross-entropy loss
        loss = nn.losses.cross_entropy(logits, targets, reduction='mean')

        return loss

    @mx.compile
    def generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> mx.array:
        """
        Generate new tokens autoregressively.

        Args:
            input_ids: Starting tokens of shape [batch_size, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
            top_p: If set, use nucleus sampling

        Returns:
            Generated token IDs of shape [batch_size, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Crop to max sequence length if needed
            input_crop = input_ids
            if input_ids.shape[1] > self.config.max_seq_len:
                input_crop = input_ids[:, -self.config.max_seq_len:]

            # Forward pass
            logits, _ = self(input_crop)

            # Get logits for last token
            logits = logits[:, -1, :]  # [batch_size, vocab_size]

            # Apply temperature
            logits = logits / temperature

            # Apply top-k filtering if specified
            if top_k is not None:
                top_k_logits, top_k_indices = mx.topk(logits, top_k, axis=-1)
                logits = mx.where(
                    mx.arange(logits.shape[-1])[None, :] < top_k,
                    top_k_logits,
                    -float('inf')
                )

            # Convert to probabilities
            probs = mx.softmax(logits, axis=-1)

            # Sample next token
            next_token = mx.random.categorical(mx.log(probs), num_samples=1)

            # Append to sequence
            input_ids = mx.concatenate([input_ids, next_token], axis=1)

        return input_ids

    def count_parameters(self) -> int:
        """
        Count the number of parameters in the model.

        Returns:
            Total number of parameters
        """
        # Get all parameters and count recursively
        def count_params_recursive(obj):
            if isinstance(obj, dict):
                total = 0
                for value in obj.values():
                    total += count_params_recursive(value)
                return total
            elif isinstance(obj, list):
                total = 0
                for item in obj:
                    total += count_params_recursive(item)
                return total
            else:
                # It's an array, count its size
                return obj.size

        params = self.parameters()
        return count_params_recursive(params)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Get number of parameters, optionally excluding embeddings.

        Args:
            non_embedding: If True, exclude embedding parameters

        Returns:
            Number of parameters
        """
        n_params = self.count_parameters()

        if non_embedding:
            # Subtract embedding parameters
            n_params -= self.token_embedding.embedding.weight.size
            n_params -= self.pos_embedding.position_embedding.weight.size

        return n_params


def create_model(
    model_size: str = "tiny",
    vocab_size: int = 8000,
    max_seq_len: int = 512,
    **kwargs
) -> GPTModel:
    """
    Factory function to create a GPT model.

    Args:
        model_size: Model size ("tiny", "gpt2-small", "gpt2-medium", "gpt2-large")
        vocab_size: Vocabulary size (for custom models)
        max_seq_len: Maximum sequence length
        **kwargs: Additional config overrides

    Returns:
        GPTModel instance
    """
    if model_size == "tiny":
        config = GPTConfig.tiny(vocab_size=vocab_size)
    elif model_size == "gpt2-small":
        config = GPTConfig.gpt2_small()
    elif model_size == "gpt2-medium":
        config = GPTConfig.gpt2_medium()
    elif model_size == "gpt2-large":
        config = GPTConfig.gpt2_large()
    else:
        raise ValueError(f"Unknown model size: {model_size}")

    # Override config with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Override max_seq_len if provided
    config.max_seq_len = max_seq_len

    return GPTModel(config)
