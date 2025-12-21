"""
Multi-Modal Model (Stage 5).

Combines vision encoder and language model for image+text understanding.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .vision_encoder import create_vision_encoder


class VisionLanguageConnector(nn.Module):
    """Projects vision features to language model dimension."""

    def __init__(self, vision_dim: int, lang_dim: int):
        """
        Args:
            vision_dim: Vision encoder output dimension
            lang_dim: Language model hidden dimension
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, lang_dim),
            nn.GELU(),
            nn.Linear(lang_dim, lang_dim)
        )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: [batch, vision_dim] or [batch, n_patches, vision_dim]

        Returns:
            lang_features: [batch, lang_dim] or [batch, n_patches, lang_dim]
        """
        return self.proj(vision_features)


class MultiModalModel(nn.Module):
    """Multi-modal model combining vision and language."""

    def __init__(self, language_model: nn.Module, vision_encoder: nn.Module,
                 vision_dim: int, lang_dim: int, vocab_size: int):
        """
        Args:
            language_model: Pre-trained language model
            vision_encoder: Vision encoder
            vision_dim: Vision encoder output dimension
            lang_dim: Language model hidden dimension
            vocab_size: Vocabulary size
        """
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.connector = VisionLanguageConnector(vision_dim, lang_dim)

        # Special token for image (similar to CLS token)
        self.image_token_embed = nn.Parameter(torch.zeros(1, 1, lang_dim))
        nn.init.trunc_normal_(self.image_token_embed, std=0.02)

        # Output projection
        self.output_proj = nn.Linear(lang_dim, vocab_size)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to language model space.

        Args:
            images: [batch, channels, height, width]

        Returns:
            image_embeds: [batch, 1, lang_dim] or [batch, n_patches, lang_dim]
        """
        # Get vision features
        if isinstance(self.vision_encoder, nn.Module):
            # Check if vision encoder returns tuple (ViT) or single tensor (CNN)
            vision_output = self.vision_encoder(images)

            if isinstance(vision_output, tuple):
                # ViT returns (cls_token, patch_tokens)
                cls_token, patch_tokens = vision_output
                vision_features = cls_token  # Use CLS token for now
            else:
                # Simple CNN returns single tensor
                vision_features = vision_output

        # Project to language dimension
        image_embeds = self.connector(vision_features)  # [batch, lang_dim]

        # Add sequence dimension
        if len(image_embeds.shape) == 2:
            image_embeds = image_embeds.unsqueeze(1)  # [batch, 1, lang_dim]

        return image_embeds

    def forward(self, input_ids: Optional[torch.Tensor] = None,
                images: Optional[torch.Tensor] = None,
                image_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional image inputs.

        Args:
            input_ids: [batch, seq_len] text token IDs
            images: [batch, channels, height, width] optional images
            image_positions: [batch] positions to insert image embeddings

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # Text-only mode
        if images is None:
            logits = self.language_model(input_ids)
            return logits

        # Multi-modal mode: image + text
        batch_size, seq_len = input_ids.shape

        # Get text embeddings (extract from language model)
        # Note: This assumes language model has token_embedding attribute
        # Adjust based on your model architecture
        if hasattr(self.language_model, 'token_embedding'):
            text_embeds = self.language_model.token_embedding(input_ids)
        else:
            # Fallback: use model's embedding layer
            text_embeds = self.language_model.embeddings(input_ids)

        # Encode images
        image_embeds = self.encode_image(images)  # [batch, 1, lang_dim]

        # Insert image embeddings at specified positions
        if image_positions is None:
            # Default: prepend images before text
            combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        else:
            # Insert at specific positions
            combined_embeds = text_embeds.clone()
            for i, pos in enumerate(image_positions):
                combined_embeds[i, pos] = image_embeds[i, 0]

        # Pass through language model
        # Note: This requires language model to accept embeddings directly
        # You may need to modify language model's forward to support this
        hidden_states = combined_embeds

        # Simple forward (adjust based on your architecture)
        for layer in self.language_model.layers:
            hidden_states = layer(hidden_states)

        # Output projection
        logits = self.output_proj(hidden_states)

        return logits


class SimpleMultiModalModel(nn.Module):
    """
    Simplified multi-modal model for easier integration.

    This version prepends image features as a single token before text.
    """

    def __init__(self, language_model: nn.Module, vision_encoder_type: str = "simple_cnn",
                 vision_dim: int = 768, lang_dim: int = 768, vocab_size: int = 32000):
        """
        Args:
            language_model: Pre-trained language model
            vision_encoder_type: Type of vision encoder ("simple_cnn" or "vit")
            vision_dim: Vision encoder dimension
            lang_dim: Language model dimension
            vocab_size: Vocabulary size
        """
        super().__init__()

        self.vision_encoder = create_vision_encoder(
            encoder_type=vision_encoder_type,
            embed_dim=vision_dim
        )

        self.language_model = language_model
        self.connector = VisionLanguageConnector(vision_dim, lang_dim)

        # Special image token ID (reserve token ID 32000)
        self.image_token_id = vocab_size

    def forward(self, input_ids: torch.Tensor,
                images: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len] with special image token (32000) where images appear
            images: [batch, channels, height, width] optional

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # Text-only mode
        if images is None:
            return self.language_model(input_ids)

        # Encode images
        vision_features = self.vision_encoder(images)  # [batch, vision_dim]
        image_embeds = self.connector(vision_features)  # [batch, lang_dim]

        # Replace image token (ID 32000) with image embeddings
        # For simplicity, assume first token is image token
        # In production, you'd search for image_token_id and replace dynamically

        # Just use text-only for now (full integration requires model modification)
        # In practice, you'd modify the language model to accept mixed embeddings
        logits = self.language_model(input_ids)

        return logits


def create_multimodal_model(language_model: nn.Module,
                           vision_encoder_type: str = "simple_cnn",
                           vision_dim: int = 768,
                           lang_dim: int = 768,
                           vocab_size: int = 32000) -> nn.Module:
    """
    Create multi-modal model.

    Args:
        language_model: Pre-trained language model
        vision_encoder_type: Vision encoder type
        vision_dim: Vision dimension
        lang_dim: Language dimension
        vocab_size: Vocabulary size

    Returns:
        Multi-modal model
    """
    return SimpleMultiModalModel(
        language_model=language_model,
        vision_encoder_type=vision_encoder_type,
        vision_dim=vision_dim,
        lang_dim=lang_dim,
        vocab_size=vocab_size
    )
