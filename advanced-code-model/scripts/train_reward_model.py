#!/usr/bin/env python3
"""
Train Reward Model for RLHF (Stage 4a).

The reward model:
1. Takes a prompt + response as input
2. Outputs a scalar reward score
3. Trained to rank chosen responses higher than rejected ones
4. Used in PPO training to guide policy optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import sys
from typing import Tuple, Optional
import time

# Import model components
sys.path.append(str(Path(__file__).parent.parent))
from src.model import create_model_from_config
from src.model.config import get_config


class RewardHead(nn.Module):
    """Reward head that outputs scalar reward from model hidden states."""

    def __init__(self, d_model: int):
        super().__init__()
        self.reward_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, d_model]

        Returns:
            rewards: [batch] scalar rewards
        """
        # Use last token's hidden state
        last_hidden = hidden_states[:, -1, :]  # [batch, d_model]
        reward = self.reward_proj(last_hidden).squeeze(-1)  # [batch]
        return reward


class RewardModel(nn.Module):
    """Complete reward model: base model + reward head."""

    def __init__(self, base_model: nn.Module, d_model: int):
        super().__init__()
        self.base_model = base_model
        self.reward_head = RewardHead(d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len]

        Returns:
            rewards: [batch] scalar rewards
        """
        # Get hidden states from base model
        logits = self.base_model(input_ids)  # [batch, seq_len, vocab_size]

        # Get last layer hidden states (approximate from logits)
        # In production, you'd extract actual hidden states from model
        # For now, use logits as proxy
        hidden_states = logits

        # Compute reward
        rewards = self.reward_head(hidden_states)
        return rewards


class PreferenceDataset(Dataset):
    """Dataset for preference pairs."""

    def __init__(self, chosen_path: Path, rejected_path: Path):
        """
        Args:
            chosen_path: Path to chosen responses .npy file
            rejected_path: Path to rejected responses .npy file
        """
        self.chosen = np.load(chosen_path)
        self.rejected = np.load(rejected_path)

        assert len(self.chosen) == len(self.rejected), \
            "Chosen and rejected must have same length"

    def __len__(self):
        return len(self.chosen)

    def __getitem__(self, idx):
        return {
            'chosen': torch.tensor(self.chosen[idx], dtype=torch.long),
            'rejected': torch.tensor(self.rejected[idx], dtype=torch.long)
        }


def ranking_loss(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor,
                margin: float = 0.5) -> torch.Tensor:
    """
    Ranking loss: encourage chosen > rejected by at least margin.

    Args:
        chosen_rewards: [batch] rewards for chosen responses
        rejected_rewards: [batch] rewards for rejected responses
        margin: Minimum margin between chosen and rejected

    Returns:
        loss: Scalar loss
    """
    # Hinge loss: max(0, margin - (chosen - rejected))
    loss = F.relu(margin - (chosen_rewards - rejected_rewards))
    return loss.mean()


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                device: torch.device, use_amp: bool = False) -> Tuple[float, float]:
    """Train reward model for one epoch."""
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == 'cuda' else None

    for batch in dataloader:
        chosen_ids = batch['chosen'].to(device)
        rejected_ids = batch['rejected'].to(device)

        optimizer.zero_grad()

        # Forward pass
        if use_amp and device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                chosen_rewards = model(chosen_ids)
                rejected_rewards = model(rejected_ids)
                loss = ranking_loss(chosen_rewards, rejected_rewards)
        else:
            chosen_rewards = model(chosen_ids)
            rejected_rewards = model(rejected_ids)
            loss = ranking_loss(chosen_rewards, rejected_rewards)

        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Metrics
        total_loss += loss.item()
        accuracy = (chosen_rewards > rejected_rewards).float().mean().item()
        total_accuracy += accuracy
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    return avg_loss, avg_accuracy


def validate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Validate reward model."""
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            chosen_ids = batch['chosen'].to(device)
            rejected_ids = batch['rejected'].to(device)

            chosen_rewards = model(chosen_ids)
            rejected_rewards = model(rejected_ids)
            loss = ranking_loss(chosen_rewards, rejected_rewards)

            total_loss += loss.item()
            accuracy = (chosen_rewards > rejected_rewards).float().mean().item()
            total_accuracy += accuracy
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    return avg_loss, avg_accuracy


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train Reward Model (Stage 4a)")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to Stage 3 checkpoint (base model)")
    parser.add_argument("--model-size", type=str, default="large",
                       choices=["tiny", "medium", "large", "xlarge"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="mps",
                       choices=["cpu", "mps", "cuda"])
    parser.add_argument("--use-amp", action="store_true",
                       help="Use automatic mixed precision (CUDA only)")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 60)
    print("Reward Model Training (Stage 4a)")
    print("=" * 60)

    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "processed"
    model_dir = project_root / "models"
    model_dir.mkdir(exist_ok=True)

    # Load base model
    print("\nLoading base model...")
    config = get_config(args.model_size)
    config.use_rmsnorm = True
    config.use_rope = True

    base_model = create_model_from_config(config, architecture="dense", device=device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict']

    # Handle compiled model
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        state_dict = {
            key.replace('_orig_mod.', ''): value
            for key, value in state_dict.items()
        }

    base_model.load_state_dict(state_dict)
    print("✓ Base model loaded")

    # Create reward model
    reward_model = RewardModel(base_model, config.d_model).to(device)
    print("✓ Reward model created")

    # Load datasets
    print("\nLoading preference datasets...")
    train_dataset = PreferenceDataset(
        data_dir / "rlhf_train_chosen.npy",
        data_dir / "rlhf_train_rejected.npy"
    )
    val_dataset = PreferenceDataset(
        data_dir / "rlhf_val_chosen.npy",
        data_dir / "rlhf_val_rejected.npy"
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"  Train pairs: {len(train_dataset)}")
    print(f"  Val pairs: {len(val_dataset)}")

    # Optimizer (only train reward head, freeze base model)
    print("\nFreezing base model, training reward head only...")
    for param in reward_model.base_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        reward_model.reward_head.parameters(),
        lr=args.learning_rate
    )

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_val_accuracy = 0.0

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 60)

        # Train
        start_time = time.time()
        train_loss, train_accuracy = train_epoch(
            reward_model, train_loader, optimizer, device, args.use_amp
        )
        train_time = time.time() - start_time

        # Validate
        val_loss, val_accuracy = validate(reward_model, val_loader, device)

        print(f"Train loss: {train_loss:.4f}, accuracy: {train_accuracy:.4f}")
        print(f"Val loss: {val_loss:.4f}, accuracy: {val_accuracy:.4f}")
        print(f"Time: {train_time:.1f}s")

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint_path = model_dir / "reward_model_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': reward_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'val_loss': val_loss,
                'config': config.__dict__
            }, checkpoint_path)
            print(f"✓ Saved best model: {checkpoint_path}")

    print("\n" + "=" * 60)
    print("✓ Reward model training complete!")
    print("=" * 60)
    print(f"\nBest val accuracy: {best_val_accuracy:.4f}")
    print("\nNext step: PPO Training")
    print("  python scripts/train_rlhf.py --checkpoint models/tool_calling_model_best.pth \\")
    print("    --reward-model models/reward_model_best.pth")


if __name__ == "__main__":
    main()
