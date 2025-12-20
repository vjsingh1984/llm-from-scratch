"""
MLX-optimized model components for code generation.
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
    MLXTransformer,
    MLXMultiHeadAttention,
    MLXFeedForward,
    MLXTransformerBlock,
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
    'MLXTransformer',
    'MLXMultiHeadAttention',
    'MLXFeedForward',
    'MLXTransformerBlock',
    'create_model',
]
