"""
Model configuration classes for production-scale code generation.

Provides pre-configured model sizes optimized for Apple Silicon:
- Tiny: 124M parameters (testing)
- Medium: 350M parameters (M1 Max)
- Large: 780M parameters (M1 Ultra)
- XLarge: 1.5B parameters (M2 Ultra)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for transformer-based code generation model."""

    # Architecture
    vocab_size: int = 32000
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    d_ff: int = 4096
    max_seq_len: int = 4096

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1

    # Training
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Generation
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9

    def __post_init__(self):
        """Validate configuration."""
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"

        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.n_layers > 0, "n_layers must be positive"

    @property
    def d_head(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads

    @property
    def num_parameters(self) -> int:
        """Approximate number of parameters in millions."""
        # Token embeddings
        token_emb = self.vocab_size * self.d_model

        # Position embeddings
        pos_emb = self.max_seq_len * self.d_model

        # Transformer blocks
        # For each layer:
        # - Attention: 4 * d_model * d_model (Q, K, V, O projections)
        # - Layer norm 1: 2 * d_model
        # - FFN: 2 * d_model * d_ff (up and down projections)
        # - Layer norm 2: 2 * d_model
        per_layer = (
            4 * self.d_model * self.d_model +  # Attention
            2 * self.d_model +                  # LN1
            2 * self.d_model * self.d_ff +      # FFN
            2 * self.d_model                    # LN2
        )
        transformer_params = self.n_layers * per_layer

        # Output layer
        output = self.d_model * self.vocab_size + 2 * self.d_model

        total = token_emb + pos_emb + transformer_params + output
        return int(total / 1e6)  # Convert to millions

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'd_ff': self.d_ff,
            'max_seq_len': self.max_seq_len,
            'dropout': self.dropout,
            'attention_dropout': self.attention_dropout,
            'residual_dropout': self.residual_dropout,
            'weight_decay': self.weight_decay,
            'grad_clip': self.grad_clip,
            'temperature': self.temperature,
            'top_k': self.top_k,
            'top_p': self.top_p,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ModelConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


def get_tiny_config() -> ModelConfig:
    """
    Tiny model configuration (124M parameters).

    Ideal for:
    - Quick testing and prototyping
    - Limited memory (8GB)
    - Fast iteration
    """
    return ModelConfig(
        vocab_size=16000,
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_ff=3072,
        max_seq_len=2048,
        dropout=0.1,
    )


def get_medium_config() -> ModelConfig:
    """
    Medium model configuration (350M parameters).

    Recommended for M1 Max:
    - Good balance of quality and speed
    - Memory: ~16GB
    - Training speed: ~35K tok/s
    - Production-ready quality
    """
    return ModelConfig(
        vocab_size=32000,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        d_ff=4096,
        max_seq_len=4096,
        dropout=0.1,
    )


def get_large_config() -> ModelConfig:
    """
    Large model configuration (780M parameters).

    Recommended for M1 Ultra:
    - High quality generation
    - Memory: ~32GB
    - Training speed: ~28K tok/s
    - Best for complex code patterns
    """
    return ModelConfig(
        vocab_size=32000,
        d_model=1536,
        n_layers=32,
        n_heads=24,
        d_ff=6144,
        max_seq_len=4096,
        dropout=0.1,
    )


def get_xlarge_config() -> ModelConfig:
    """
    XLarge model configuration (1.5B parameters).

    Recommended for M2 Ultra:
    - Maximum quality
    - Memory: ~48GB
    - Training speed: ~22K tok/s
    - State-of-the-art performance
    """
    return ModelConfig(
        vocab_size=50000,
        d_model=2048,
        n_layers=40,
        n_heads=32,
        d_ff=8192,
        max_seq_len=4096,
        dropout=0.1,
    )


def get_config(size: str = 'medium') -> ModelConfig:
    """
    Get model configuration by size name.

    Args:
        size: One of 'tiny', 'medium', 'large', 'xlarge'

    Returns:
        ModelConfig instance

    Raises:
        ValueError: If size is not recognized
    """
    configs = {
        'tiny': get_tiny_config,
        'medium': get_medium_config,
        'large': get_large_config,
        'xlarge': get_xlarge_config,
    }

    if size not in configs:
        raise ValueError(
            f"Unknown config size: {size}. "
            f"Choose from: {list(configs.keys())}"
        )

    config = configs[size]()

    print(f"Model Configuration: {size.upper()}")
    print(f"  Parameters: {config.num_parameters}M")
    print(f"  Layers: {config.n_layers}")
    print(f"  Hidden dim: {config.d_model}")
    print(f"  Attention heads: {config.n_heads}")
    print(f"  FFN dim: {config.d_ff}")
    print(f"  Max sequence: {config.max_seq_len}")
    print(f"  Vocabulary: {config.vocab_size}")

    return config


if __name__ == '__main__':
    """Test configuration classes."""
    print("=" * 60)
    print("Model Configurations for Apple Silicon")
    print("=" * 60)
    print()

    for size in ['tiny', 'medium', 'large', 'xlarge']:
        config = get_config(size)
        print()
        print("-" * 60)
        print()
