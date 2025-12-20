"""
Mixture of Experts (MoE) Transformer.

Based on:
- "Switch Transformers" (Fedus et al., 2021): https://arxiv.org/abs/2101.03961
- "Mixtral 8x7B" (Mistral AI, 2023)

Key advantages:
- Sparse activation: only use subset of parameters per token
- Can scale to 10x+ parameters with 2x compute
- Specialist sub-models for different input types
- Better parameter efficiency

Architecture:
- Standard attention layers
- FFN replaced with MoE layer (multiple expert networks)
- Router network decides which experts to use
- Load balancing to ensure all experts are used
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from .config import ModelConfig
from .transformer import MultiHeadAttention, RMSNorm


class Expert(nn.Module):
    """Single expert network (FFN)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through expert.

        Args:
            x: Input [batch, seq_len, d_model]

        Returns:
            Output [batch, seq_len, d_model]
        """
        return self.dropout(self.w2(F.gelu(self.w1(x))))


class Router(nn.Module):
    """
    Router network for MoE.

    Decides which experts should process each token.
    Includes load balancing to ensure all experts are utilized.
    """

    def __init__(self, d_model: int, num_experts: int, expert_capacity: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity

        # Router weights
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.

        Args:
            x: Input [batch, seq_len, d_model]

        Returns:
            - expert_weights: Routing weights [batch, seq_len, expert_capacity]
            - expert_indices: Which experts to use [batch, seq_len, expert_capacity]
            - load_balancing_loss: Auxiliary loss for load balancing
        """
        batch, seq_len, d_model = x.shape

        # Compute routing scores
        router_logits = self.gate(x)  # [batch, seq_len, num_experts]

        # Top-K routing (pick top expert_capacity experts)
        expert_weights, expert_indices = torch.topk(
            router_logits, k=self.expert_capacity, dim=-1
        )  # [batch, seq_len, expert_capacity]

        # Normalize weights (softmax over selected experts)
        expert_weights = F.softmax(expert_weights, dim=-1)

        # Load balancing loss (encourage equal expert usage)
        # Fraction of tokens routed to each expert
        router_probs = F.softmax(router_logits, dim=-1)  # [batch, seq_len, num_experts]
        expert_usage = router_probs.mean(dim=[0, 1])  # [num_experts]

        # Ideal usage is 1/num_experts for each expert
        ideal_usage = 1.0 / self.num_experts
        load_balancing_loss = self.num_experts * torch.sum(expert_usage ** 2) - 1.0

        return expert_weights, expert_indices, load_balancing_loss


class MoELayer(nn.Module):
    """
    Mixture of Experts layer.

    Replaces standard FFN with multiple expert networks and routing.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.expert_capacity = config.expert_capacity

        # Create experts
        self.experts = nn.ModuleList([
            Expert(config.d_model, config.d_ff, config.dropout)
            for _ in range(self.num_experts)
        ])

        # Router
        self.router = Router(config.d_model, self.num_experts, self.expert_capacity)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE layer.

        Args:
            x: Input [batch, seq_len, d_model]

        Returns:
            - output: [batch, seq_len, d_model]
            - load_balancing_loss: Auxiliary loss for load balancing
        """
        batch, seq_len, d_model = x.shape

        # Route tokens to experts
        expert_weights, expert_indices, load_balancing_loss = self.router(x)
        # expert_weights: [batch, seq_len, expert_capacity]
        # expert_indices: [batch, seq_len, expert_capacity]

        # Initialize output
        output = torch.zeros_like(x)

        # Process each token through its selected experts
        for i in range(self.expert_capacity):
            # Get expert indices for this capacity slot
            expert_idx = expert_indices[:, :, i]  # [batch, seq_len]
            weights = expert_weights[:, :, i].unsqueeze(-1)  # [batch, seq_len, 1]

            # Process through each expert
            for expert_id in range(self.num_experts):
                # Mask for tokens routed to this expert
                mask = (expert_idx == expert_id).unsqueeze(-1)  # [batch, seq_len, 1]

                if mask.any():
                    # Process tokens through expert
                    expert_out = self.experts[expert_id](x)  # [batch, seq_len, d_model]

                    # Add weighted output (only for tokens routed to this expert)
                    output = output + mask * weights * expert_out

        return output, load_balancing_loss


class MoETransformerBlock(nn.Module):
    """
    Transformer block with Mixture of Experts.

    Uses standard attention but replaces FFN with MoE layer.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Normalization
        if config.use_rmsnorm:
            self.ln1 = RMSNorm(config.d_model)
            self.ln2 = RMSNorm(config.d_model)
        else:
            self.ln1 = nn.LayerNorm(config.d_model)
            self.ln2 = nn.LayerNorm(config.d_model)

        # Attention (standard multi-head attention)
        self.attention = MultiHeadAttention(config)

        # MoE layer (replaces standard FFN)
        self.moe = MoELayer(config)

        self.dropout = nn.Dropout(config.residual_dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE transformer block.

        Args:
            x: Input [batch, seq_len, d_model]
            mask: Attention mask [batch, seq_len, seq_len]

        Returns:
            - output: [batch, seq_len, d_model]
            - load_balancing_loss: Auxiliary loss
        """
        # Attention with residual
        if self.config.use_gradient_checkpointing and self.training:
            attn_out = torch.utils.checkpoint.checkpoint(
                lambda x_in: self.attention(self.ln1(x_in), mask=mask),
                x,
                use_reentrant=False
            )
        else:
            attn_out = self.attention(self.ln1(x), mask=mask)

        x = x + self.dropout(attn_out)

        # MoE with residual
        moe_out, load_balancing_loss = self.moe(self.ln2(x))
        x = x + self.dropout(moe_out)

        return x, load_balancing_loss


class MoETransformer(nn.Module):
    """
    Mixture of Experts Transformer.

    Sparse model that activates only a subset of parameters per token.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token + position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        if not config.use_rope:
            self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # MoE transformer blocks
        self.blocks = nn.ModuleList([
            MoETransformerBlock(config)
            for _ in range(config.n_layers)
        ])

        # Final normalization
        if config.use_rmsnorm:
            self.ln_f = RMSNorm(config.d_model)
        else:
            self.ln_f = nn.LayerNorm(config.d_model)

        # Output projection (tie weights with input embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Weight tying

        self.dropout = nn.Dropout(config.dropout)

        # Initialize weights
        self.apply(self._init_weights)

        # Calculate parameters (note: MoE has more params but fewer active)
        total_params = self.count_parameters()
        active_params = total_params / config.num_experts * config.expert_capacity

        print(f"Initialized MoE Transformer:")
        print(f"  Architecture: Mixture of Experts (Sparse)")
        print(f"  Total parameters: {total_params / 1e6:.1f}M")
        print(f"  Active per token: {active_params / 1e6:.1f}M ({config.expert_capacity}/{config.num_experts} experts)")
        print(f"  Layers: {config.n_layers}")
        print(f"  Hidden dim: {config.d_model}")
        print(f"  Attention heads: {config.n_heads}")
        print(f"  Experts: {config.num_experts}")
        print(f"  Expert capacity: {config.expert_capacity}")
        print(f"  Vocabulary: {config.vocab_size}")

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input token IDs [batch, seq_len]

        Returns:
            Logits [batch, seq_len, vocab_size]

        Note: Load balancing loss is computed but not returned (used in training loop)
        """
        batch, seq_len = x.shape
        device = x.device

        # Token embeddings
        tok_emb = self.token_embedding(x)

        # Position embeddings (if not using RoPE)
        if not self.config.use_rope:
            pos = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
            pos_emb = self.position_embedding(pos)
            x = tok_emb + pos_emb
        else:
            x = tok_emb

        x = self.dropout(x)

        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)

        # MoE transformer blocks
        total_load_balancing_loss = 0.0
        for block in self.blocks:
            x, load_balancing_loss = block(x, mask=mask)
            total_load_balancing_loss = total_load_balancing_loss + load_balancing_loss

        # Store load balancing loss for training (access via model.last_load_balancing_loss)
        self.last_load_balancing_loss = total_load_balancing_loss / len(self.blocks)

        # Final norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits


def create_moe_model(config: ModelConfig, device: str = "cpu") -> MoETransformer:
    """
    Create MoE model and move to device.

    Args:
        config: Model configuration
        device: Device to use ('cpu', 'cuda', 'mps')

    Returns:
        Initialized MoE model
    """
    model = MoETransformer(config)
    model = model.to(device)
    return model
