"""
Optimizer configuration for PyTorch training.

Includes:
- AdamW optimizer
- Learning rate schedulers
- Gradient clipping
"""

import torch
import math
from typing import Optional


def configure_optimizer(
    model,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.95),
    eps: float = 1e-8
):
    """
    Configure AdamW optimizer.

    Args:
        model: PyTorch model
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        betas: Adam beta parameters
        eps: Adam epsilon

    Returns:
        Optimizer instance
    """
    # Separate parameters that should/shouldn't have weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # No weight decay for biases and LayerNorm parameters
        if 'bias' in name or 'ln' in name or 'norm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=betas,
        eps=eps
    )

    print(f"Optimizer configured:")
    print(f"  Parameters with decay: {len(decay_params)}")
    print(f"  Parameters without decay: {len(no_decay_params)}")

    return optimizer


def get_lr_scheduler(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    scheduler_type: str = 'cosine',
    min_lr_ratio: float = 0.1
):
    """
    Get learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        scheduler_type: Type of scheduler ('cosine', 'linear', or 'constant')
        min_lr_ratio: Minimum LR as ratio of initial LR

    Returns:
        LR scheduler
    """
    if scheduler_type == 'cosine':
        def lr_lambda(current_step: int):
            # Warmup
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))

            # Cosine decay
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

            # Scale to [min_lr_ratio, 1.0]
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    elif scheduler_type == 'linear':
        def lr_lambda(current_step: int):
            # Warmup
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))

            # Linear decay
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(min_lr_ratio, 1.0 - progress)

    elif scheduler_type == 'constant':
        def lr_lambda(current_step: int):
            # Warmup only
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return 1.0

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"LR Scheduler configured:")
    print(f"  Type: {scheduler_type}")
    print(f"  Warmup steps: {num_warmup_steps}")
    print(f"  Total steps: {num_training_steps}")
    print(f"  Min LR ratio: {min_lr_ratio}")

    return scheduler


def clip_grad_norm_(
    parameters,
    max_norm: float,
    norm_type: float = 2.0
) -> float:
    """
    Clip gradient norm (wrapper around torch function).

    Args:
        parameters: Model parameters
        max_norm: Max norm
        norm_type: Type of norm

    Returns:
        Total gradient norm
    """
    return torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm,
        norm_type=norm_type
    )


def test_optimizer():
    """Test optimizer configuration."""
    print("Testing Optimizer Configuration")
    print("="*60)

    # Create dummy model
    from model import create_model

    model = create_model('tiny', vocab_size=100, device='cpu')

    # Configure optimizer
    optimizer = configure_optimizer(model, learning_rate=3e-4)

    # Create scheduler
    scheduler = get_lr_scheduler(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=1000,
        scheduler_type='cosine'
    )

    print(f"\nInitial LR: {optimizer.param_groups[0]['lr']}")

    # Simulate steps
    lrs = []
    for step in range(1000):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    print(f"LR after 50 steps (warmup): {lrs[50]:.6f}")
    print(f"LR after 100 steps (peak): {lrs[100]:.6f}")
    print(f"LR after 500 steps (mid): {lrs[500]:.6f}")
    print(f"LR after 999 steps (end): {lrs[999]:.6f}")


if __name__ == '__main__':
    test_optimizer()
