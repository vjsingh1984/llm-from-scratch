"""
Training loop implementation for language models.

Handles training, validation, checkpointing, and logging.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable
import time
import json

from .data_loader import DataLoader
from .optimizer import get_lr_schedule, clip_gradients


@dataclass
class TrainerConfig:
    """
    Configuration for model training.
    """
    # Training hyperparameters
    max_epochs: int = 10
    max_steps: Optional[int] = None  # If set, overrides max_epochs
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8

    # Learning rate schedule
    warmup_steps: int = 2000
    lr_schedule_type: str = "cosine"  # "cosine", "linear", "constant"
    min_lr: float = 0.0

    # Optimization
    gradient_clip: float = 1.0
    grad_accumulation_steps: int = 1

    # Logging and checkpointing
    log_interval: int = 10  # Log every N steps
    eval_interval: int = 500  # Evaluate every N steps
    save_interval: int = 1000  # Save checkpoint every N steps
    checkpoint_dir: str = "checkpoints"

    # Generation during evaluation
    generate_samples: bool = True
    num_samples: int = 3
    max_gen_tokens: int = 50

    def __post_init__(self):
        """Validate configuration."""
        if self.max_steps is None and self.max_epochs is None:
            raise ValueError("Either max_steps or max_epochs must be set")


class Trainer:
    """
    Trainer for language models.

    Handles training loop, optimization, evaluation, and checkpointing.

    Args:
        model: Language model to train
        config: Training configuration
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        tokenizer: Tokenizer for text generation (optional)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        tokenizer = None
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # Calculate total steps
        if config.max_steps:
            self.total_steps = config.max_steps
        else:
            self.total_steps = len(train_loader) * config.max_epochs

        # Create optimizer and LR schedule
        self.optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )

        self.lr_schedule = get_lr_schedule(
            schedule_type=config.lr_schedule_type,
            max_lr=config.learning_rate,
            warmup_steps=config.warmup_steps,
            total_steps=self.total_steps,
            min_lr=config.min_lr
        )

        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training stats
        self.stats = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'steps': [],
        }

    def loss_fn(self, inputs: mx.array, targets: mx.array):
        """
        Compute loss for a batch.

        Args:
            inputs: Input token IDs [batch_size, seq_len]
            targets: Target token IDs [batch_size, seq_len]

        Returns:
            Loss value
        """
        logits, loss = self.model(inputs, targets)
        return loss

    def train_step(self, inputs: mx.array, targets: mx.array) -> tuple[float, float]:
        """
        Single training step.

        Args:
            inputs: Input tokens
            targets: Target tokens

        Returns:
            Tuple of (loss, gradient_norm)
        """
        # Forward and backward pass
        loss_and_grad_fn = nn.value_and_grad(self.model, self.loss_fn)
        loss, grads = loss_and_grad_fn(inputs, targets)

        # Clip gradients
        grads, grad_norm = clip_gradients(grads, self.config.gradient_clip)

        # Update model parameters
        self.optimizer.update(self.model, grads)

        # Force computation
        mx.eval(self.model.parameters(), self.optimizer.state)

        return loss.item(), grad_norm

    def evaluate(self) -> float:
        """
        Evaluate model on validation set.

        Returns:
            Average validation loss
        """
        if self.val_loader is None:
            return float('nan')

        total_loss = 0.0
        num_batches = 0

        for inputs, targets in self.val_loader:
            logits, loss = self.model(inputs, targets)
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
        return avg_loss

    def generate_samples(self, num_samples: int = 3, max_tokens: int = 50) -> list[str]:
        """
        Generate text samples for monitoring.

        Args:
            num_samples: Number of samples to generate
            max_tokens: Maximum tokens per sample

        Returns:
            List of generated text strings
        """
        if self.tokenizer is None:
            return []

        samples = []
        prompts = [
            "Once upon a time",
            "The cat",
            "In a galaxy"
        ][:num_samples]

        for prompt in prompts:
            # Encode prompt
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = mx.array([prompt_ids])

            # Generate
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=0.8
            )

            # Decode
            generated_text = self.tokenizer.decode(
                output_ids[0].tolist(),
                skip_special_tokens=True
            )
            samples.append(generated_text)

        return samples

    def save_checkpoint(self, filename: str = None):
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename (default: step_{step}.npz)
        """
        if filename is None:
            filename = f"step_{self.step}.npz"

        checkpoint_path = self.checkpoint_dir / filename

        # Save model weights
        self.model.save_weights(str(checkpoint_path))

        # Save training state
        state_path = checkpoint_path.with_suffix('.json')
        state = {
            'step': self.step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'config': {
                'learning_rate': self.config.learning_rate,
                'total_steps': self.total_steps,
            }
        }

        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, filename: str):
        """
        Load model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = self.checkpoint_dir / filename

        # Load model weights
        self.model.load_weights(str(checkpoint_path))

        # Load training state
        state_path = checkpoint_path.with_suffix('.json')
        if state_path.exists():
            with open(state_path, 'r') as f:
                state = json.load(f)

            self.step = state['step']
            self.epoch = state['epoch']
            self.best_val_loss = state['best_val_loss']

        print(f"Checkpoint loaded: {checkpoint_path}")

    def train(self):
        """
        Main training loop.
        """
        print("="*60)
        print("Starting Training")
        print("="*60)
        print(f"Total steps: {self.total_steps}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("="*60)

        start_time = time.time()
        running_loss = 0.0

        while True:
            # Check if done
            if self.config.max_steps and self.step >= self.config.max_steps:
                break

            # Epoch loop
            for inputs, targets in self.train_loader:
                # Training step
                loss, grad_norm = self.train_step(inputs, targets)
                running_loss += loss

                # Update learning rate
                current_lr = self.lr_schedule(self.step)
                self.optimizer.learning_rate = current_lr

                self.step += 1

                # Logging
                if self.step % self.config.log_interval == 0:
                    avg_loss = running_loss / self.config.log_interval
                    elapsed = time.time() - start_time
                    tokens_per_sec = (self.config.log_interval *
                                    inputs.shape[0] * inputs.shape[1]) / elapsed

                    print(
                        f"Step {self.step}/{self.total_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Grad Norm: {grad_norm:.2f} | "
                        f"Tokens/s: {tokens_per_sec:.0f}"
                    )

                    self.stats['train_losses'].append(avg_loss)
                    self.stats['learning_rates'].append(current_lr)
                    self.stats['steps'].append(self.step)

                    running_loss = 0.0
                    start_time = time.time()

                # Evaluation
                if self.step % self.config.eval_interval == 0:
                    print("\nEvaluating...")
                    val_loss = self.evaluate()
                    print(f"Validation Loss: {val_loss:.4f}")

                    self.stats['val_losses'].append({
                        'step': self.step,
                        'loss': val_loss
                    })

                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best.npz")
                        print("New best model saved!")

                    # Generate samples
                    if self.config.generate_samples:
                        try:
                            print("\nGenerated samples:")
                            samples = self.generate_samples(
                                self.config.num_samples,
                                self.config.max_gen_tokens
                            )
                            for i, sample in enumerate(samples):
                                print(f"  {i+1}. {sample[:100]}...")
                            print()
                        except Exception as e:
                            print(f"Generation failed: {e}")

                # Checkpointing
                if self.step % self.config.save_interval == 0:
                    self.save_checkpoint()

                # Check if done
                if self.config.max_steps and self.step >= self.config.max_steps:
                    break

            self.epoch += 1

            # Check if done (by epochs)
            if not self.config.max_steps and self.config.max_epochs:
                if self.epoch >= self.config.max_epochs:
                    break

        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Total steps: {self.step}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        # Save final checkpoint
        self.save_checkpoint("final.npz")

        # Save training stats
        stats_path = self.checkpoint_dir / "training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"Training stats saved: {stats_path}")
