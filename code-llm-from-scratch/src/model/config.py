"""
Model configuration for bash code generation.

Smaller models optimized for code generation task.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CoderConfig:
    """Configuration for code generation model."""

    # Model architecture
    vocab_size: int = 8000  # Code vocabularies are smaller
    n_layers: int = 6
    d_model: int = 384
    n_heads: int = 6
    d_ff: int = 1536  # 4 * d_model
    max_seq_len: int = 512

    # Regularization
    dropout: float = 0.1
    attn_dropout: float = 0.1

    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_steps: int = 10000

    # Code-specific
    use_bias: bool = True
    layer_norm_eps: float = 1e-5

    # Generation
    temperature: float = 0.7  # Lower for code (less random)
    top_k: int = 40
    top_p: float = 0.9

    @classmethod
    def tiny_coder(cls, vocab_size: int = 8000):
        """Tiny coder model (~12M parameters)."""
        return cls(
            vocab_size=vocab_size,
            n_layers=6,
            d_model=384,
            n_heads=6,
            d_ff=1536,
            max_seq_len=512
        )

    @classmethod
    def small_coder(cls, vocab_size: int = 8000):
        """Small coder model (~50M parameters)."""
        return cls(
            vocab_size=vocab_size,
            n_layers=12,
            d_model=768,
            n_heads=12,
            d_ff=3072,
            max_seq_len=1024
        )

    @classmethod
    def medium_coder(cls, vocab_size: int = 8000):
        """Medium coder model (~150M parameters)."""
        return cls(
            vocab_size=vocab_size,
            n_layers=18,
            d_model=1024,
            n_heads=16,
            d_ff=4096,
            max_seq_len=1024
        )

    def get_num_parameters(self) -> int:
        """
        Estimate number of parameters.

        Rough calculation:
        - Embeddings: vocab_size * d_model + max_seq_len * d_model
        - Each layer: 12 * d_model^2 (approximately)
        - Total: embeddings + n_layers * 12 * d_model^2
        """
        embed_params = self.vocab_size * self.d_model + self.max_seq_len * self.d_model
        layer_params = 12 * self.d_model * self.d_model
        total = embed_params + self.n_layers * layer_params
        return total


# Pre-configured model sizes with parameter counts
MODEL_CONFIGS = {
    'tiny': (CoderConfig.tiny_coder(), '~12M params'),
    'small': (CoderConfig.small_coder(), '~50M params'),
    'medium': (CoderConfig.medium_coder(), '~150M params'),
}


def get_model_config(size: str = 'tiny', vocab_size: int = 8000) -> CoderConfig:
    """
    Get predefined model configuration.

    Args:
        size: Model size ('tiny', 'small', 'medium')
        vocab_size: Vocabulary size

    Returns:
        CoderConfig instance
    """
    if size == 'tiny':
        return CoderConfig.tiny_coder(vocab_size)
    elif size == 'small':
        return CoderConfig.small_coder(vocab_size)
    elif size == 'medium':
        return CoderConfig.medium_coder(vocab_size)
    else:
        raise ValueError(f"Unknown model size: {size}")
