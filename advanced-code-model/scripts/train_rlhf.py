#!/usr/bin/env python3
"""
PPO Training for RLHF (Stage 4b).

Uses Proximal Policy Optimization to fine-tune the model based on reward signals:
1. Generate responses from current policy
2. Compute rewards using reward model
3. Update policy using PPO objective
4. Repeat until convergence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import sys
from typing import Dict, List, Tuple, Optional
import time
from tokenizers import Tokenizer

# Import model components
sys.path.append(str(Path(__file__).parent.parent))
from src.model import create_model_from_config
from src.model.config import get_config
from train_reward_model import RewardModel


class RLHFDataset(Dataset):
    """Dataset of prompts for RL training."""

    def __init__(self, prompts: List[str], tokenizer: Tokenizer, max_length: int = 128):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        formatted = f"<|user|>{prompt}<|end|>\n<|assistant|>"

        # Tokenize
        encoding = self.tokenizer.encode(formatted)
        ids = encoding.ids

        # Truncate or pad
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
        else:
            ids = ids + [0] * (self.max_length - len(ids))

        return torch.tensor(ids, dtype=torch.long)


def generate_response(model: nn.Module, prompt_ids: torch.Tensor, tokenizer: Tokenizer,
                     max_new_tokens: int = 256, temperature: float = 0.8) -> torch.Tensor:
    """
    Generate response from model.

    Args:
        model: Policy model
        prompt_ids: [batch, seq_len] prompt token IDs
        tokenizer: Tokenizer
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        response_ids: [batch, response_len] generated token IDs
    """
    model.eval()
    batch_size = prompt_ids.shape[0]
    device = prompt_ids.device

    with torch.no_grad():
        output_ids = prompt_ids.clone()

        for _ in range(max_new_tokens):
            logits = model(output_ids)
            next_token_logits = logits[:, -1, :] / temperature

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            output_ids = torch.cat([output_ids, next_token], dim=1)

            # Check for end token
            end_token_id = tokenizer.token_to_id("<|end|>")
            if end_token_id is not None:
                if (next_token == end_token_id).all():
                    break

    # Extract response (everything after prompt)
    response_ids = output_ids[:, prompt_ids.shape[1]:]
    return response_ids


def compute_advantages(rewards: torch.Tensor, values: torch.Tensor,
                      gamma: float = 0.99, lambda_: float = 0.95) -> torch.Tensor:
    """
    Compute GAE (Generalized Advantage Estimation).

    Args:
        rewards: [batch] rewards
        values: [batch] value estimates
        gamma: Discount factor
        lambda_: GAE lambda

    Returns:
        advantages: [batch] advantage estimates
    """
    # Simplified advantage computation (no temporal dimension for now)
    # In full RL, you'd compute advantages over time steps
    advantages = rewards - values
    return advantages


def ppo_loss(logits: torch.Tensor, old_logits: torch.Tensor, actions: torch.Tensor,
            advantages: torch.Tensor, epsilon: float = 0.2) -> torch.Tensor:
    """
    Compute PPO clipped objective.

    Args:
        logits: [batch, seq_len, vocab_size] new policy logits
        old_logits: [batch, seq_len, vocab_size] old policy logits (detached)
        actions: [batch, seq_len] sampled actions
        advantages: [batch] advantage estimates
        epsilon: Clipping epsilon

    Returns:
        loss: PPO loss
    """
    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    old_log_probs = F.log_softmax(old_logits, dim=-1)

    # Gather log probs for taken actions
    action_log_probs = torch.gather(log_probs, -1, actions.unsqueeze(-1)).squeeze(-1)
    old_action_log_probs = torch.gather(old_log_probs, -1, actions.unsqueeze(-1)).squeeze(-1)

    # Average over sequence
    action_log_probs = action_log_probs.mean(dim=1)  # [batch]
    old_action_log_probs = old_action_log_probs.mean(dim=1)  # [batch]

    # Compute ratio
    ratio = torch.exp(action_log_probs - old_action_log_probs)

    # Expand advantages for broadcasting
    advantages = advantages.unsqueeze(1).expand_as(ratio)

    # PPO clipped objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    loss = -torch.min(surr1, surr2).mean()

    return loss


def train_ppo_step(policy_model: nn.Module, reward_model: nn.Module,
                  prompts: torch.Tensor, tokenizer: Tokenizer,
                  optimizer: torch.optim.Optimizer, device: torch.device) -> Dict[str, float]:
    """
    Single PPO training step.

    Args:
        policy_model: Policy model to train
        reward_model: Reward model (frozen)
        prompts: [batch, prompt_len] prompt token IDs
        tokenizer: Tokenizer
        optimizer: Optimizer
        device: Device

    Returns:
        metrics: Dictionary of training metrics
    """
    policy_model.train()
    reward_model.eval()

    # Generate responses from current policy
    with torch.no_grad():
        responses = generate_response(policy_model, prompts, tokenizer, max_new_tokens=128)

    # Combine prompts and responses
    full_sequences = torch.cat([prompts, responses], dim=1)

    # Compute rewards
    with torch.no_grad():
        rewards = reward_model(full_sequences)

    # Get old policy logits (for PPO ratio)
    with torch.no_grad():
        old_logits = policy_model(full_sequences)

    # Compute values (use mean reward as value estimate)
    values = rewards.mean() * torch.ones_like(rewards)

    # Compute advantages
    advantages = compute_advantages(rewards, values)

    # Forward pass with new policy
    new_logits = policy_model(full_sequences)

    # Compute PPO loss
    loss = ppo_loss(new_logits, old_logits, full_sequences, advantages)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
    optimizer.step()

    # Metrics
    metrics = {
        'loss': loss.item(),
        'mean_reward': rewards.mean().item(),
        'mean_advantage': advantages.mean().item(),
    }

    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PPO Training for RLHF (Stage 4b)")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to Stage 3 checkpoint (policy model)")
    parser.add_argument("--reward-model", type=str, required=True,
                       help="Path to trained reward model")
    parser.add_argument("--model-size", type=str, default="large",
                       choices=["tiny", "medium", "large", "xlarge"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="mps",
                       choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 60)
    print("PPO Training for RLHF (Stage 4b)")
    print("=" * 60)

    # Paths
    project_root = Path(__file__).parent.parent
    tokenizer_path = project_root / "data" / "tokenizer" / "tokenizer.json"
    model_dir = project_root / "models"

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print("✓ Tokenizer loaded")

    # Load policy model
    print("\nLoading policy model...")
    config = get_config(args.model_size)
    config.use_rmsnorm = True
    config.use_rope = True

    policy_model = create_model_from_config(config, architecture="dense", device=device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict']

    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        state_dict = {
            key.replace('_orig_mod.', ''): value
            for key, value in state_dict.items()
        }

    policy_model.load_state_dict(state_dict)
    print("✓ Policy model loaded")

    # Load reward model
    print("\nLoading reward model...")
    base_model = create_model_from_config(config, architecture="dense", device=device)
    reward_model = RewardModel(base_model, config.d_model).to(device)

    reward_checkpoint = torch.load(args.reward_model, map_location=device)
    reward_state_dict = reward_checkpoint['model_state_dict']

    reward_model.load_state_dict(reward_state_dict)
    reward_model.eval()
    print("✓ Reward model loaded")

    # Create dataset of prompts
    prompts_text = [
        "Write a function to check if a number is prime",
        "Implement binary search algorithm",
        "Create a function to reverse a linked list",
        "Write a merge sort implementation",
        "Implement a stack data structure",
    ]

    dataset = RLHFDataset(prompts_text, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print(f"\nDataset size: {len(dataset)}")

    # Optimizer
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.learning_rate)

    # Training loop
    print("\n" + "=" * 60)
    print("Starting PPO training...")
    print("=" * 60)

    global_step = 0
    best_reward = float('-inf')

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 60)

        epoch_metrics = []

        for step in range(args.steps_per_epoch):
            # Sample batch
            for batch in dataloader:
                prompts = batch.to(device)

                # PPO step
                metrics = train_ppo_step(
                    policy_model, reward_model, prompts, tokenizer, optimizer, device
                )

                epoch_metrics.append(metrics)
                global_step += 1

                if step % 20 == 0:
                    avg_reward = np.mean([m['mean_reward'] for m in epoch_metrics[-20:]])
                    avg_loss = np.mean([m['loss'] for m in epoch_metrics[-20:]])
                    print(f"Step {step}/{args.steps_per_epoch} | "
                          f"Reward: {avg_reward:.4f} | Loss: {avg_loss:.4f}")

                break  # One batch per step

        # Epoch summary
        avg_reward = np.mean([m['mean_reward'] for m in epoch_metrics])
        avg_loss = np.mean([m['loss'] for m in epoch_metrics])

        print(f"\nEpoch {epoch + 1} complete:")
        print(f"  Avg reward: {avg_reward:.4f}")
        print(f"  Avg loss: {avg_loss:.4f}")

        # Save best model
        if avg_reward > best_reward:
            best_reward = avg_reward
            checkpoint_path = model_dir / "rlhf_model_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': policy_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'reward': avg_reward,
                'config': config.__dict__
            }, checkpoint_path)
            print(f"✓ Saved best model: {checkpoint_path}")

    print("\n" + "=" * 60)
    print("✓ RLHF training complete!")
    print("=" * 60)
    print(f"\nBest reward: {best_reward:.4f}")
    print("\nNext stage: Multi-Modal (Stage 5)")


if __name__ == "__main__":
    main()
