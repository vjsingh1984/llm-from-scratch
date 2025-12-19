"""
Optimizer utilities and learning rate schedules.

Implements AdamW optimizer and various LR schedules for transformer training.
"""

import mlx.core as mx
import mlx.optimizers as optim
import math
from typing import Callable, Optional


def create_optimizer(
    model,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.95),
    eps: float = 1e-8,
    optimizer_type: str = "adamw"
) -> optim.Optimizer:
    """
    Create optimizer for model training.

    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay (L2 regularization)
        betas: Betas for Adam optimizer
        eps: Epsilon for numerical stability
        optimizer_type: Type of optimizer ("adamw", "adam", "sgd")

    Returns:
        MLX optimizer instance
    """
    if optimizer_type == "adamw":
        return optim.AdamW(
            learning_rate=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    elif optimizer_type == "adam":
        return optim.Adam(
            learning_rate=learning_rate,
            betas=betas,
            eps=eps
        )
    elif optimizer_type == "sgd":
        return optim.SGD(
            learning_rate=learning_rate,
            momentum=betas[0]
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def get_cosine_schedule_with_warmup(
    max_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 0.0
) -> Callable[[int], float]:
    """
    Create cosine learning rate schedule with warmup.

    Learning rate increases linearly during warmup, then follows cosine decay.

    Args:
        max_lr: Maximum learning rate (reached after warmup)
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr: Minimum learning rate at the end

    Returns:
        Function that takes step number and returns learning rate
    """
    def lr_schedule(step: int) -> float:
        # Warmup phase: linear increase
        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps

        # Cosine decay phase
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]

        # Cosine annealing
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        lr = min_lr + (max_lr - min_lr) * cosine_decay

        return lr

    return lr_schedule


def get_linear_schedule_with_warmup(
    max_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 0.0
) -> Callable[[int], float]:
    """
    Create linear learning rate schedule with warmup.

    LR increases during warmup, then decreases linearly.

    Args:
        max_lr: Maximum learning rate
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr: Minimum LR at the end

    Returns:
        LR schedule function
    """
    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps

        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        progress = max(0.0, min(1.0, progress))

        lr = max_lr - (max_lr - min_lr) * progress
        return lr

    return lr_schedule


def get_constant_schedule_with_warmup(
    max_lr: float,
    warmup_steps: int
) -> Callable[[int], float]:
    """
    Constant LR after warmup.

    Args:
        max_lr: Learning rate after warmup
        warmup_steps: Number of warmup steps

    Returns:
        LR schedule function
    """
    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps
        return max_lr

    return lr_schedule


def get_lr_schedule(
    schedule_type: str,
    max_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 0.0
) -> Callable[[int], float]:
    """
    Factory function to create LR schedule.

    Args:
        schedule_type: Type of schedule ("cosine", "linear", "constant")
        max_lr: Maximum learning rate
        warmup_steps: Warmup steps
        total_steps: Total training steps
        min_lr: Minimum learning rate

    Returns:
        LR schedule function
    """
    if schedule_type == "cosine":
        return get_cosine_schedule_with_warmup(max_lr, warmup_steps, total_steps, min_lr)
    elif schedule_type == "linear":
        return get_linear_schedule_with_warmup(max_lr, warmup_steps, total_steps, min_lr)
    elif schedule_type == "constant":
        return get_constant_schedule_with_warmup(max_lr, warmup_steps)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def clip_gradients(grads: dict, max_norm: float) -> tuple:
    """
    Clip gradients by global norm.

    Args:
        grads: Dictionary of gradients (possibly nested)
        max_norm: Maximum norm for clipping

    Returns:
        Tuple of (clipped_gradients, total_norm)
    """
    # Compute global norm recursively
    def compute_norm_recursive(obj):
        if isinstance(obj, dict):
            total = 0.0
            for value in obj.values():
                total += compute_norm_recursive(value)
            return total
        elif isinstance(obj, list):
            total = 0.0
            for item in obj:
                total += compute_norm_recursive(item)
            return total
        elif obj is not None:
            # It's an array
            return mx.sum(obj * obj).item()
        else:
            return 0.0

    # Clip gradients recursively
    def clip_recursive(obj, clip_coef):
        if isinstance(obj, dict):
            return {k: clip_recursive(v, clip_coef) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clip_recursive(item, clip_coef) for item in obj]
        elif obj is not None:
            return obj * clip_coef
        else:
            return None

    total_norm = math.sqrt(compute_norm_recursive(grads))

    # Clip if necessary
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        grads = clip_recursive(grads, clip_coef)

    return grads, total_norm


class GradientAccumulator:
    """
    Accumulate gradients over multiple mini-batches.

    Useful for training with larger effective batch sizes on limited memory.

    Args:
        accumulation_steps: Number of steps to accumulate gradients
    """

    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.accumulated_grads = None
        self.current_step = 0

    def accumulate(self, grads: dict) -> bool:
        """
        Accumulate gradients.

        Args:
            grads: Current gradients

        Returns:
            True if should perform optimizer step, False otherwise
        """
        if self.accumulated_grads is None:
            # Initialize accumulated gradients
            self.accumulated_grads = {k: mx.zeros_like(v) for k, v in grads.items()}

        # Add current gradients
        for key in grads:
            self.accumulated_grads[key] = self.accumulated_grads[key] + grads[key]

        self.current_step += 1

        # Check if should perform optimizer step
        if self.current_step >= self.accumulation_steps:
            return True
        return False

    def get_and_reset(self) -> dict:
        """
        Get accumulated gradients and reset.

        Returns:
            Accumulated gradients (averaged)
        """
        # Average gradients
        avg_grads = {k: v / self.accumulation_steps
                     for k, v in self.accumulated_grads.items()}

        # Reset
        self.accumulated_grads = None
        self.current_step = 0

        return avg_grads


def configure_optimizers(
    model,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.95),
    eps: float = 1e-8,
    warmup_steps: int = 2000,
    total_steps: int = 100000,
    schedule_type: str = "cosine",
    min_lr: float = 0.0
) -> tuple[optim.Optimizer, Callable]:
    """
    Configure optimizer and learning rate schedule.

    Convenience function that creates both optimizer and LR schedule.

    Args:
        model: Model to optimize
        learning_rate: Peak learning rate
        weight_decay: Weight decay
        betas: Adam betas
        eps: Adam epsilon
        warmup_steps: Warmup steps
        total_steps: Total training steps
        schedule_type: LR schedule type
        min_lr: Minimum learning rate

    Returns:
        Tuple of (optimizer, lr_schedule_fn)
    """
    # Create optimizer
    optimizer = create_optimizer(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps
    )

    # Create LR schedule
    lr_schedule = get_lr_schedule(
        schedule_type=schedule_type,
        max_lr=learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=min_lr
    )

    return optimizer, lr_schedule
