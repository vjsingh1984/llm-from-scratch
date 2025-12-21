"""
Mamba: Linear-Time Sequence Modeling with Selective State Spaces.

Based on "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
by Gu & Dao (2023): https://arxiv.org/abs/2312.00752

Key advantages over Transformers:
- O(n) complexity instead of O(n²) attention
- Constant memory regardless of sequence length
- Better scaling to very long sequences
- Selective state space allows content-aware reasoning

Architecture:
- Selective SSM blocks instead of attention
- No position embeddings needed (implicit in SSM)
- Gated activation for better expressivity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

from .config import ModelConfig


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (S6) layer.

    This is the core of Mamba - a selective SSM that can efficiently model
    long-range dependencies with linear complexity.

    Optimizations:
    - Parallel scan algorithm (vs sequential loop)
    - Chunked processing for memory efficiency
    - Vectorized operations
    """

    def __init__(self, d_model: int, state_size: int, conv_size: int = 4,
                 use_parallel_scan: bool = False, chunk_size: int = 256):  # Sequential for stability
        super().__init__()
        self.d_model = d_model
        self.state_size = state_size
        self.conv_size = conv_size
        self.use_parallel_scan = use_parallel_scan
        self.chunk_size = chunk_size

        # Project input to SSM parameters (B, C, Δ)
        self.x_proj = nn.Linear(d_model, state_size + state_size + 1, bias=False)

        # Input projection (d_model -> state_size) for B
        self.in_proj = nn.Linear(d_model, state_size, bias=False)

        # Output projection (state_size -> d_model) for C
        self.out_proj = nn.Linear(state_size, d_model, bias=False)

        # Convolution for short-range dependencies
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=conv_size,
            padding=conv_size - 1,
            groups=d_model,  # Depthwise convolution
        )

        # SSM parameters (A is state transition matrix)
        # Initialize A with special structure for stability
        A = torch.arange(1, state_size + 1).float()
        self.register_buffer("A_log", torch.log(A))

        # D is skip connection (important for stability)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through selective SSM.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = x.shape

        # 1. Convolution for short-range dependencies
        # [batch, seq_len, d_model] -> [batch, d_model, seq_len]
        x_conv = x.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Trim padding
        x_conv = x_conv.transpose(1, 2)  # Back to [batch, seq_len, d_model]

        # 2. Get SSM parameters (selective based on input)
        ssm_params = self.x_proj(x)  # [batch, seq_len, state_size*2 + 1]
        B, C, delta = torch.split(
            ssm_params,
            [self.state_size, self.state_size, 1],
            dim=-1
        )
        delta = F.softplus(delta)  # Ensure positive step size

        # 3. Discretize continuous SSM (zero-order hold)
        A = -torch.exp(self.A_log.float())  # [state_size]
        # Discretization: A_discrete = exp(A * delta)
        deltaA = torch.exp(A.unsqueeze(0).unsqueeze(0) * delta)  # [batch, seq_len, state_size]
        deltaB = delta * B  # [batch, seq_len, state_size]

        # 4. Run SSM (scan over sequence)
        # Project input to state space
        x_proj_state = self.in_proj(x_conv)  # [batch, seq_len, state_size]

        # Choose parallel or sequential scan
        if self.use_parallel_scan:
            # Parallel scan (FAST!) - O(log n) parallel depth
            h_states = self._parallel_scan(deltaA, deltaB, x_proj_state)
        else:
            # Sequential scan (SLOW) - O(n) sequential steps
            h_states = self._sequential_scan(deltaA, deltaB, x_proj_state)

        # Compute outputs from states
        # h_states: [batch, seq_len, state_size]
        # C: [batch, seq_len, state_size]
        h_modulated = C * h_states  # [batch, seq_len, state_size]

        # Project to output space and add skip connection
        output = self.out_proj(h_modulated) + self.D.unsqueeze(0).unsqueeze(0) * x_conv
        # output: [batch, seq_len, d_model]

        return output

    def _parallel_scan(self, deltaA: torch.Tensor, deltaB: torch.Tensor,
                      x: torch.Tensor) -> torch.Tensor:
        """
        Parallel associative scan for SSM.

        Computes h_t = A_t * h_{t-1} + B_t * x_t for all t in parallel.

        This is the KEY optimization that makes Mamba fast!

        Args:
            deltaA: [batch, seq_len, state_size]
            deltaB: [batch, seq_len, state_size]
            x: [batch, seq_len, state_size]

        Returns:
            h_states: [batch, seq_len, state_size]
        """
        batch, seq_len, state_size = x.shape

        # Binary associative scan (parallel prefix sum)
        # Works by combining pairs at each level of binary tree
        # O(log n) parallel depth instead of O(n) sequential

        # Initialize: h_0 = 0, so first state is just B_0 * x_0
        # Represent each element as (A, Bu) pair for associative operation
        # (A2, Bu2) ⊗ (A1, Bu1) = (A2 * A1, A2 * Bu1 + Bu2)

        # Compute Bu (element-wise product B * u where u=x)
        Bu = deltaB * x  # [batch, seq_len, state_size]

        # Use chunks to avoid memory issues with very long sequences
        if seq_len > self.chunk_size:
            return self._chunked_parallel_scan(deltaA, Bu, seq_len, batch, state_size)

        # For shorter sequences, use direct parallel scan
        return self._direct_parallel_scan(deltaA, Bu, batch, state_size)

    def _direct_parallel_scan(self, A: torch.Tensor, Bu: torch.Tensor,
                              batch: int, state_size: int) -> torch.Tensor:
        """
        Parallel scan using binary tree reduction - O(log n) depth.

        For recurrence h_t = A_t * h_{t-1} + Bu_t, we use associative operator:
        (A2, Bu2) ⊗ (A1, Bu1) = (A2 * A1, A2 * Bu1 + Bu2)

        This allows parallel prefix sum in O(log n) steps.

        Args:
            A: [batch, seq_len, state_size] - transition matrices
            Bu: [batch, seq_len, state_size] - B * x terms

        Returns:
            h_states: [batch, seq_len, state_size]
        """
        batch, seq_len, state_size = A.shape

        # Pad to power of 2 for binary tree
        next_pow2 = 2 ** math.ceil(math.log2(max(seq_len, 2)))
        if seq_len < next_pow2:
            pad_len = next_pow2 - seq_len
            A = F.pad(A, (0, 0, 0, pad_len), value=1.0)  # Identity for A
            Bu = F.pad(Bu, (0, 0, 0, pad_len), value=0.0)  # Zero for Bu

        # Store as pairs (A, Bu) for associative scan
        A_scan = A.clone()  # [batch, next_pow2, state_size]
        Bu_scan = Bu.clone()  # [batch, next_pow2, state_size]

        num_levels = int(math.log2(next_pow2))

        # Up-sweep: build binary tree (avoid in-place for autograd)
        for level in range(num_levels):
            stride = 2 ** (level + 1)
            # Create new tensors to avoid in-place modifications
            A_new = A_scan.clone()
            Bu_new = Bu_scan.clone()

            # Update positions at stride, 2*stride, 3*stride, ...
            for i in range(stride - 1, next_pow2, stride):
                # Combine with left sibling
                left_i = i - 2 ** level
                # (A[i], Bu[i]) = (A[i], Bu[i]) ⊗ (A[left_i], Bu[left_i])
                # = (A[i] * A[left_i], A[i] * Bu[left_i] + Bu[i])
                Bu_new[:, i] = A_scan[:, i] * Bu_scan[:, left_i] + Bu_scan[:, i]
                A_new[:, i] = A_scan[:, i] * A_scan[:, left_i]

            A_scan = A_new
            Bu_scan = Bu_new

        # Down-sweep: propagate results down the tree
        # Create new tensors for root modification
        A_new = A_scan.clone()
        Bu_new = Bu_scan.clone()
        A_new[:, -1] = 1.0
        Bu_new[:, -1] = 0.0
        A_scan = A_new
        Bu_scan = Bu_new

        for level in range(num_levels - 1, -1, -1):
            stride = 2 ** (level + 1)
            A_new = A_scan.clone()
            Bu_new = Bu_scan.clone()

            for i in range(stride - 1, next_pow2, stride):
                left_i = i - 2 ** level
                # Swap and combine (no in-place)
                temp_A = A_scan[:, i]
                temp_Bu = Bu_scan[:, i]

                # Right child gets parent value
                A_new[:, i] = temp_A * A_scan[:, left_i]
                Bu_new[:, i] = temp_A * Bu_scan[:, left_i] + temp_Bu

                # Left child gets original parent value
                A_new[:, left_i] = temp_A
                Bu_new[:, left_i] = temp_Bu

            A_scan = A_new
            Bu_scan = Bu_new

        # The scan result is in Bu_scan
        # But we need to compute h from the scan - actually Bu_scan already contains h!
        # Because h[t] = scan result at position t

        # Remove padding
        h_states = Bu_scan[:, :seq_len]

        return h_states

    def _chunked_parallel_scan(self, A: torch.Tensor, Bu: torch.Tensor,
                                seq_len: int, batch: int, state_size: int) -> torch.Tensor:
        """
        Chunked parallel scan for very long sequences.

        Process sequence in chunks, then combine chunk results.

        Args:
            A: [batch, seq_len, state_size]
            Bu: [batch, seq_len, state_size]

        Returns:
            h_states: [batch, seq_len, state_size]
        """
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        h_states = torch.zeros(batch, seq_len, state_size, device=A.device, dtype=A.dtype)
        h_prev = torch.zeros(batch, state_size, device=A.device, dtype=A.dtype)

        for chunk_idx in range(num_chunks):
            start = chunk_idx * self.chunk_size
            end = min((chunk_idx + 1) * self.chunk_size, seq_len)
            chunk_len = end - start

            # Get chunk
            A_chunk = A[:, start:end]  # [batch, chunk_len, state_size]
            Bu_chunk = Bu[:, start:end]

            # Scan within chunk (can use sequential for small chunks or parallel for large)
            if chunk_len <= 64:
                # Small chunk: use fast sequential
                h_chunk = self._sequential_scan_chunk(A_chunk, Bu_chunk, h_prev)
            else:
                # Large chunk: use parallel scan
                h_chunk = self._direct_parallel_scan(A_chunk, Bu_chunk, batch, state_size)
                # Adjust for initial state
                if chunk_idx > 0:
                    # Propagate h_prev through chunk
                    A_cum = torch.cumprod(A_chunk, dim=1)  # [batch, chunk_len, state_size]
                    h_chunk = A_cum * h_prev.unsqueeze(1) + h_chunk

            h_states[:, start:end] = h_chunk

            # Update h_prev for next chunk (last state of current chunk)
            h_prev = h_chunk[:, -1]

        return h_states

    def _sequential_scan_chunk(self, A: torch.Tensor, Bu: torch.Tensor,
                               h_init: torch.Tensor) -> torch.Tensor:
        """
        Sequential scan for a small chunk.

        Args:
            A: [batch, chunk_len, state_size]
            Bu: [batch, chunk_len, state_size]
            h_init: [batch, state_size]

        Returns:
            h_states: [batch, chunk_len, state_size]
        """
        batch, chunk_len, state_size = A.shape
        h_states = torch.zeros_like(A)
        h = h_init

        for t in range(chunk_len):
            h = A[:, t] * h + Bu[:, t]
            h_states[:, t] = h

        return h_states

    def _sequential_scan(self, deltaA: torch.Tensor, deltaB: torch.Tensor,
                        x: torch.Tensor) -> torch.Tensor:
        """
        Sequential scan (fallback, slow).

        Args:
            deltaA: [batch, seq_len, state_size]
            deltaB: [batch, seq_len, state_size]
            x: [batch, seq_len, state_size]

        Returns:
            h_states: [batch, seq_len, state_size]
        """
        batch, seq_len, state_size = x.shape
        h_states = torch.zeros_like(x)
        h = torch.zeros(batch, state_size, device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            h = deltaA[:, t] * h + deltaB[:, t] * x[:, t]
            h_states[:, t] = h

        return h_states


class MambaBlock(nn.Module):
    """
    Mamba block: SSM + MLP with gating.

    Similar to Transformer block but uses SSM instead of attention.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Normalization (use RMSNorm by default for Mamba)
        if config.use_rmsnorm:
            from .transformer import RMSNorm
            self.norm = RMSNorm(config.d_model)
        else:
            self.norm = nn.LayerNorm(config.d_model)

        # Selective SSM
        self.ssm = SelectiveSSM(
            d_model=config.d_model,
            state_size=config.state_size,
            conv_size=config.conv_size
        )

        # Gated MLP (like SwiGLU but for Mamba)
        self.mlp_gate = nn.Linear(config.d_model, config.d_ff)
        self.mlp_up = nn.Linear(config.d_model, config.d_ff)
        self.mlp_down = nn.Linear(config.d_ff, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Mamba block.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Pre-normalization
        x_norm = self.norm(x)

        # SSM
        ssm_out = self.ssm(x_norm)

        # Residual
        x = x + self.dropout(ssm_out)

        # Gated MLP
        x_norm = self.norm(x)
        gate = F.silu(self.mlp_gate(x_norm))  # SiLU activation
        up = self.mlp_up(x_norm)
        mlp_out = self.mlp_down(gate * up)

        # Residual
        x = x + self.dropout(mlp_out)

        return x


class MambaModel(nn.Module):
    """
    Mamba language model.

    Uses selective state space models instead of attention for O(n) complexity.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings (no position embeddings needed for SSM!)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(config)
            for _ in range(config.n_layers)
        ])

        # Final normalization
        if config.use_rmsnorm:
            from .transformer import RMSNorm
            self.ln_f = RMSNorm(config.d_model)
        else:
            self.ln_f = nn.LayerNorm(config.d_model)

        # Output projection (tie weights with input embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Weight tying

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Initialize weights
        self.apply(self._init_weights)

        print(f"Initialized Mamba Model:")
        print(f"  Architecture: Selective State Space (O(n) complexity)")
        print(f"  Parameters: {self.count_parameters() / 1e6:.1f}M")
        print(f"  Layers: {config.n_layers}")
        print(f"  Hidden dim: {config.d_model}")
        print(f"  State size: {config.state_size}")
        print(f"  FFN dim: {config.d_ff}")
        print(f"  Vocabulary: {config.vocab_size}")
        print(f"  No position embeddings (implicit in SSM)")

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
        """
        # Token embeddings
        x = self.token_embedding(x)
        x = self.dropout(x)

        # Mamba blocks (no attention mask needed!)
        for block in self.blocks:
            if self.config.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, use_reentrant=False
                )
            else:
                x = block(x)

        # Final norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits


def create_mamba_model(config: ModelConfig, device: str = "cpu") -> MambaModel:
    """
    Create Mamba model and move to device.

    Args:
        config: Model configuration
        device: Device to use ('cpu', 'cuda', 'mps')

    Returns:
        Initialized Mamba model
    """
    model = MambaModel(config)
    model = model.to(device)
    return model
