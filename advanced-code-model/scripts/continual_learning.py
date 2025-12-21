#!/usr/bin/env python3
"""
Continual Learning System (Stage 10).

Enables the model to learn from new data over time without forgetting:
- Online learning from user feedback
- Replay buffer for catastrophic forgetting prevention
- Adaptive learning rates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
from typing import List, Dict, Tuple
import numpy as np
from collections import deque

sys.path.append(str(Path(__file__).parent.parent))
from src.model import create_model_from_config
from src.model.config import get_config


class ReplayBuffer:
    """Buffer for storing past examples to prevent catastrophic forgetting."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, input_ids: torch.Tensor, target_ids: torch.Tensor):
        """Add example to buffer."""
        self.buffer.append({
            'input_ids': input_ids.cpu(),
            'target_ids': target_ids.cpu()
        })

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample batch from buffer."""
        if len(self.buffer) == 0:
            return None, None

        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)

        input_ids = torch.stack([self.buffer[i]['input_ids'] for i in indices])
        target_ids = torch.stack([self.buffer[i]['target_ids'] for i in indices])

        return input_ids, target_ids

    def __len__(self):
        return len(self.buffer)


class ContinualLearner:
    """Manages continual learning with replay and EWC."""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 device: str = "mps", replay_ratio: float = 0.5):
        """
        Args:
            model: Model to train
            optimizer: Optimizer
            device: Device
            replay_ratio: Ratio of replay examples vs new examples
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.replay_ratio = replay_ratio
        self.replay_buffer = ReplayBuffer()

        # Fisher information for EWC (Elastic Weight Consolidation)
        self.fisher_dict = {}
        self.optimal_params = {}

    def update(self, input_ids: torch.Tensor, target_ids: torch.Tensor,
              ewc_lambda: float = 0.5) -> float:
        """
        Update model with new example.

        Args:
            input_ids: New input
            target_ids: New target
            ewc_lambda: EWC regularization strength

        Returns:
            loss: Training loss
        """
        self.model.train()

        # Add to replay buffer
        self.replay_buffer.add(input_ids, target_ids)

        # Sample replay examples
        batch_size = input_ids.shape[0]
        replay_size = int(batch_size * self.replay_ratio)

        if replay_size > 0 and len(self.replay_buffer) > 0:
            replay_input, replay_target = self.replay_buffer.sample(replay_size)

            if replay_input is not None:
                # Combine new and replay
                input_ids = torch.cat([input_ids, replay_input.to(self.device)], dim=0)
                target_ids = torch.cat([target_ids, replay_target.to(self.device)], dim=0)

        # Forward pass
        logits = self.model(input_ids)

        # Compute loss (next token prediction)
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            target_ids[:, 1:].reshape(-1),
            ignore_index=0
        )

        # Add EWC regularization
        if self.fisher_dict and ewc_lambda > 0:
            ewc_loss = self._compute_ewc_loss()
            loss = loss + ewc_lambda * ewc_loss

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _compute_ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss."""
        ewc_loss = 0.0

        for name, param in self.model.named_parameters():
            if name in self.fisher_dict:
                fisher = self.fisher_dict[name]
                optimal = self.optimal_params[name]
                ewc_loss += (fisher * (param - optimal) ** 2).sum()

        return ewc_loss

    def consolidate(self):
        """Consolidate current parameters (for EWC)."""
        print("Consolidating parameters for EWC...")

        # Store current optimal parameters
        for name, param in self.model.named_parameters():
            self.optimal_params[name] = param.data.clone()

        # Compute Fisher information (simplified: use gradient magnitude)
        # In full EWC, you'd compute Fisher over validation set
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.fisher_dict[name] = param.grad.data.clone() ** 2
            else:
                self.fisher_dict[name] = torch.zeros_like(param.data)

        print("✓ Consolidation complete")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Continual Learning (Stage 10)")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to initial checkpoint")
    parser.add_argument("--model-size", type=str, default="large",
                       choices=["tiny", "medium", "large", "xlarge"])
    parser.add_argument("--device", type=str, default="mps",
                       choices=["cpu", "mps", "cuda"])
    parser.add_argument("--learning-rate", type=float, default=1e-6,
                       help="Learning rate (very low for continual learning)")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 60)
    print("Continual Learning System (Stage 10)")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    config = get_config(args.model_size)
    config.use_rmsnorm = True
    config.use_rope = True

    model = create_model_from_config(config, architecture="dense", device=device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict']

    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        state_dict = {
            key.replace('_orig_mod.', ''): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict)
    print("✓ Model loaded")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Create continual learner
    learner = ContinualLearner(model, optimizer, device=args.device)

    print("\n" + "=" * 60)
    print("Continual Learning System Ready")
    print("=" * 60)
    print(f"Replay buffer size: {learner.replay_buffer.max_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("\nThe system is ready to learn from new examples over time.")
    print("Use learner.update(input_ids, target_ids) to add new examples.")


if __name__ == "__main__":
    main()
