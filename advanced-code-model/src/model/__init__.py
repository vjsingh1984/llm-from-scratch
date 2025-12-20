"""
PyTorch model components for code generation with MPS backend.
"""

from .config import (
    ModelConfig,
    get_tiny_config,
    get_medium_config,
    get_large_config,
    get_xlarge_config,
    get_config,
)

from .transformer import (
    Transformer,
    MultiHeadAttention,
    FeedForward,
    TransformerBlock,
    RMSNorm,
    create_model,
)

__all__ = [
    # Config
    'ModelConfig',
    'get_tiny_config',
    'get_medium_config',
    'get_large_config',
    'get_xlarge_config',
    'get_config',
    # Model
    'Transformer',
    'MultiHeadAttention',
    'FeedForward',
    'TransformerBlock',
    'RMSNorm',
    'create_model',
]
