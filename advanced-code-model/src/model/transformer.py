"""
MLX-optimized Transformer model for code generation.

Key optimizations for Apple Silicon:
1. Unified memory architecture - seamless CPU-GPU data sharing
2. Lazy evaluation - computation only when needed
3. Native Metal acceleration - optimized for M1/M2/M3
4. Efficient attention - MLX-specific implementations
"""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig


class MLXMultiHeadAttention(nn.Module):
    """
    Multi-head self-attention optimized for MLX.

    Uses MLX primitives for maximum efficiency on Apple Silicon.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_head

        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(
            self.d_model,
            3 * self.d_model,
            bias=True
        )

        # Output projection
        self.out_proj = nn.Linear(self.d_model, self.d_model)

        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)

        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.d_head)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None
    ) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask [batch, seq_len, seq_len]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_head)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # [3, batch, n_heads, seq_len, d_head]

        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        # scores = (Q @ K.T) / sqrt(d_head)
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        # Apply causal mask (prevent attending to future tokens)
        if mask is None:
            # Create causal mask
            mask = mx.triu(
                mx.full((seq_len, seq_len), -1e9),
                k=1
            )

        scores = scores + mask

        # Softmax and dropout
        attn_weights = mx.softmax(scores, axis=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        out = attn_weights @ v  # [batch, n_heads, seq_len, d_head]

        # Reshape and project output
        out = out.transpose(0, 2, 1, 3)  # [batch, seq_len, n_heads, d_head]
        out = out.reshape(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out


class MLXFeedForward(nn.Module):
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

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # First linear transformation + GELU
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.dropout(x)

        # Second linear transformation
        x = self.fc2(x)
        x = self.dropout(x)

        return x


class MLXTransformerBlock(nn.Module):
    """
    Single transformer block with pre-normalization.

    Architecture:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Pre-normalization
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

        # Main components
        self.attention = MLXMultiHeadAttention(config)
        self.ffn = MLXFeedForward(config)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None
    ) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Attention block with residual connection
        x = x + self.attention(self.ln1(x), mask=mask)

        # Feed-forward block with residual connection
        x = x + self.ffn(self.ln2(x))

        return x


class MLXTransformer(nn.Module):
    """
    Complete transformer model for code generation.

    Optimized for MLX and Apple Silicon:
    - Efficient memory usage with unified architecture
    - Fast attention computation with Metal acceleration
    - Lazy evaluation for optimal performance
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model
        )

        self.position_embedding = nn.Embedding(
            config.max_seq_len,
            config.d_model
        )

        # Embedding dropout
        self.emb_dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = [
            MLXTransformerBlock(config)
            for _ in range(config.n_layers)
        ]

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)

        # Language modeling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie embeddings (share weights between input and output embeddings)
        self.lm_head.weight = self.token_embedding.weight

        print(f"Initialized MLX Transformer:")
        print(f"  Parameters: {self.num_parameters() / 1e6:.1f}M")
        print(f"  Layers: {config.n_layers}")
        print(f"  Attention heads: {config.n_heads}")

    def __call__(
        self,
        input_ids: mx.array,
        mask: Optional[mx.array] = None
    ) -> mx.array:
        """
        Forward pass.

        Args:
            input_ids: Token indices [batch, seq_len]
            mask: Optional attention mask

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        token_emb = self.token_embedding(input_ids)

        # Position embeddings
        positions = mx.arange(seq_len)
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

    def generate(
        self,
        prompt_ids: mx.array,
        max_length: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> mx.array:
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
        batch_size = prompt_ids.shape[0]
        generated = prompt_ids

        for _ in range(max_length):
            # Get logits for next token
            logits = self(generated)
            next_token_logits = logits[:, -1, :]  # [batch, vocab_size]

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_vals = mx.topk(next_token_logits, top_k, axis=-1)
                threshold = top_k_vals[:, -1:]
                next_token_logits = mx.where(
                    next_token_logits < threshold,
                    -float('inf'),
                    next_token_logits
                )

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits = mx.sort(next_token_logits, axis=-1)[:, ::-1]
                sorted_probs = mx.softmax(sorted_logits, axis=-1)
                cumsum_probs = mx.cumsum(sorted_probs, axis=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1]
                sorted_indices_to_remove[:, 0] = False

                # Apply mask
                next_token_logits = mx.where(
                    sorted_indices_to_remove,
                    -float('inf'),
                    next_token_logits
                )

            # Sample from distribution
            probs = mx.softmax(next_token_logits, axis=-1)
            next_token = mx.random.categorical(mx.log(probs), axis=-1)

            # Append to generated sequence
            next_token = next_token[:, None]  # [batch, 1]
            generated = mx.concatenate([generated, next_token], axis=1)

            # Check for end of sequence
            # (In practice, you'd check for EOS token)

        return generated

    def num_parameters(self) -> int:
        """Count total number of parameters."""
        total = 0

        # Token and position embeddings
        total += self.token_embedding.weight.size
        total += self.position_embedding.weight.size

        # Transformer blocks
        for block in self.blocks:
            # Attention
            total += block.attention.qkv_proj.weight.size
            total += block.attention.qkv_proj.bias.size
            total += block.attention.out_proj.weight.size
            total += block.attention.out_proj.bias.size

            # FFN
            total += block.ffn.fc1.weight.size
            total += block.ffn.fc1.bias.size
            total += block.ffn.fc2.weight.size
            total += block.ffn.fc2.bias.size

            # Layer norms
            total += block.ln1.weight.size
            total += block.ln1.bias.size
            total += block.ln2.weight.size
            total += block.ln2.bias.size

        # Final layer norm
        total += self.ln_f.weight.size
        total += self.ln_f.bias.size

        # LM head shares weights with token embedding, so don't double count

        return total


def create_model(config: ModelConfig) -> MLXTransformer:
    """
    Create and initialize MLX transformer model.

    Args:
        config: Model configuration

    Returns:
        Initialized MLXTransformer
    """
    model = MLXTransformer(config)

    # MLX modules are automatically initialized with reasonable defaults
    # No need for custom initialization in most cases

    return model


if __name__ == '__main__':
    """Test model creation and forward pass."""
    from .config import get_medium_config

    print("Testing MLX Transformer Model")
    print("=" * 60)

    # Create config
    config = get_medium_config()

    # Create model
    model = create_model(config)

    print()
    print("Model created successfully!")
    print(f"Total parameters: {model.num_parameters() / 1e6:.1f}M")

    # Test forward pass
    batch_size = 2
    seq_len = 128
    input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))

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
