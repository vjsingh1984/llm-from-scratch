"""
Vision Encoder for Multi-Modal Model (Stage 5).

Encodes images into feature vectors that can be processed by the language model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PatchEmbedding(nn.Module):
    """Convert image into patches and embed them."""

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 768):
        """
        Args:
            img_size: Input image size (assumes square)
            patch_size: Size of each patch
            in_channels: Number of input channels (3 for RGB)
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Conv2d acts as patch extraction + linear projection
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, height, width]

        Returns:
            patches: [batch, n_patches, embed_dim]
        """
        # Extract and project patches
        x = self.proj(x)  # [batch, embed_dim, h/p, w/p]

        # Flatten spatial dimensions
        batch, embed_dim, h, w = x.shape
        x = x.flatten(2)  # [batch, embed_dim, n_patches]
        x = x.transpose(1, 2)  # [batch, n_patches, embed_dim]

        return x


class VisionTransformer(nn.Module):
    """Vision Transformer encoder (simplified)."""

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 768,
                 depth: int = 6, n_heads: int = 12):
        """
        Args:
            img_size: Input image size
            patch_size: Patch size
            in_channels: Input channels
            embed_dim: Embedding dimension
            depth: Number of transformer layers
            n_heads: Number of attention heads
        """
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, channels, height, width]

        Returns:
            cls_token: [batch, embed_dim] - global image representation
            patch_tokens: [batch, n_patches, embed_dim] - patch features
        """
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # [batch, n_patches, embed_dim]

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [batch, n_patches+1, embed_dim]

        # Add position embeddings
        x = x + self.pos_embed

        # Transformer encoder
        x = self.transformer(x)

        # Normalize
        x = self.norm(x)

        # Split CLS token and patch tokens
        cls_token = x[:, 0]  # [batch, embed_dim]
        patch_tokens = x[:, 1:]  # [batch, n_patches, embed_dim]

        return cls_token, patch_tokens


class SimpleCNN(nn.Module):
    """Simple CNN vision encoder (lightweight alternative to ViT)."""

    def __init__(self, img_size: int = 224, in_channels: int = 3,
                 embed_dim: int = 768):
        """
        Args:
            img_size: Input image size
            in_channels: Input channels
            embed_dim: Output embedding dimension
        """
        super().__init__()

        self.conv_layers = nn.Sequential(
            # Layer 1: 224x224 -> 112x112
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Layer 2: 112x112 -> 56x56
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Layer 3: 56x56 -> 28x28
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Layer 4: 28x28 -> 14x14
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Layer 5: 14x14 -> 7x7
            nn.Conv2d(512, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Projection to embed_dim
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, height, width]

        Returns:
            features: [batch, embed_dim]
        """
        # Convolutional features
        x = self.conv_layers(x)  # [batch, embed_dim, 7, 7]

        # Global pooling
        x = self.pool(x)  # [batch, embed_dim, 1, 1]
        x = x.flatten(1)  # [batch, embed_dim]

        # Project
        x = self.proj(x)  # [batch, embed_dim]

        return x


def create_vision_encoder(encoder_type: str = "simple_cnn",
                         img_size: int = 224,
                         embed_dim: int = 768) -> nn.Module:
    """
    Create vision encoder.

    Args:
        encoder_type: "simple_cnn" or "vit"
        img_size: Input image size
        embed_dim: Output embedding dimension

    Returns:
        Vision encoder module
    """
    if encoder_type == "simple_cnn":
        return SimpleCNN(img_size=img_size, embed_dim=embed_dim)
    elif encoder_type == "vit":
        return VisionTransformer(img_size=img_size, embed_dim=embed_dim)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
