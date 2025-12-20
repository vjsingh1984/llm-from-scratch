"""
PyTorch trainer for code generation model.

Handles:
- Training loop with MPS acceleration
- Validation
- Checkpointing
- Logging
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional
import time
import json


class CodeTrainer:
    """
    Trainer for code generation models.

    PyTorch implementation optimized for M1 Max MPS backend.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device: str = 'mps',
        gradient_clip: float = 1.0,
        eval_interval: int = 100,
        checkpoint_dir: Optional[Path] = None,
        generate_samples: bool = True
    ):
        """
        Initialize trainer.

        Args:
            model: CodeTransformer model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on ('mps', 'cuda', or 'cpu')
            gradient_clip: Maximum gradient norm
            eval_interval: Steps between evaluations
            checkpoint_dir: Directory for checkpoints
            generate_samples: Whether to generate sample code during eval
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gradient_clip = gradient_clip
        self.eval_interval = eval_interval
        self.checkpoint_dir = checkpoint_dir
        self.generate_samples = generate_samples

        # Move model to device
        self.model = self.model.to(device)

        # Create checkpoint dir
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Metrics
        self.step = 0
        self.best_val_loss = float('inf')

        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Gradient clip: {gradient_clip}")
        print(f"  Eval interval: {eval_interval}")

    def train_step(self, batch) -> dict:
        """
        Single training step.

        Args:
            batch: Batch of data

        Returns:
            Dictionary with metrics
        """
        self.model.train()

        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        target_ids = batch['target_ids'].to(self.device)

        # Forward pass
        logits, loss = self.model(input_ids, target_ids)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.gradient_clip
        )

        # Update weights
        self.optimizer.step()
        self.scheduler.step()

        return {
            'loss': loss.item(),
            'grad_norm': grad_norm.item(),
            'lr': self.scheduler.get_last_lr()[0]
        }

    @torch.no_grad()
    def evaluate(self) -> dict:
        """
        Evaluate on validation set.

        Returns:
            Dictionary with metrics
        """
        self.model.eval()

        total_loss = 0
        num_batches = 0

        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)

            logits, loss = self.model(input_ids, target_ids)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        return {'val_loss': avg_loss}

    @torch.no_grad()
    def generate_sample(self, prompt: str, tokenizer, max_tokens: int = 50) -> str:
        """
        Generate sample code.

        Args:
            prompt: Starting prompt
            tokenizer: Tokenizer instance
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        self.model.eval()

        # Encode prompt
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], device=self.device)

        # Generate
        try:
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=0.8,
                top_k=40
            )

            # Decode
            generated_text = tokenizer.decode(
                output_ids[0].cpu().tolist(),
                skip_special_tokens=True
            )

            return generated_text

        except Exception as e:
            return f"Generation failed: {e}"

    def train(
        self,
        num_steps: int,
        tokenizer=None,
        log_interval: int = 10
    ):
        """
        Main training loop.

        Args:
            num_steps: Number of training steps
            tokenizer: Tokenizer for sample generation
            log_interval: Steps between logging
        """
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)

        start_time = time.time()
        train_iter = iter(self.train_loader)

        for step in range(num_steps):
            self.step = step

            # Get batch (cycle through dataloader)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            # Training step
            metrics = self.train_step(batch)

            # Log
            if step % log_interval == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = (step + 1) * batch['input_ids'].size(0) * batch['input_ids'].size(1) / elapsed

                print(f"Step {step:5d} | "
                      f"Loss: {metrics['loss']:.4f} | "
                      f"LR: {metrics['lr']:.2e} | "
                      f"Grad: {metrics['grad_norm']:.2f} | "
                      f"Tok/s: {tokens_per_sec:,.0f}")

            # Evaluate
            if step % self.eval_interval == 0 and step > 0:
                print("\n" + "-"*60)
                print(f"Evaluation at step {step}")
                print("-"*60)

                eval_metrics = self.evaluate()
                print(f"Validation Loss: {eval_metrics['val_loss']:.4f}")

                # Generate samples
                if self.generate_samples and tokenizer is not None:
                    print("\nGenerated samples:")
                    prompts = [
                        "#!/bin/bash\n",
                        "for i in",
                        "if ["
                    ]

                    for prompt in prompts:
                        generated = self.generate_sample(prompt, tokenizer, max_tokens=30)
                        print(f"\nPrompt: {repr(prompt)}")
                        print(f"Generated:\n{generated}")

                # Save checkpoint if best
                if eval_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = eval_metrics['val_loss']
                    if self.checkpoint_dir:
                        self.save_checkpoint(f"best_model.pt")
                        print(f"✓ Saved best model (val_loss={self.best_val_loss:.4f})")

                print("-"*60 + "\n")

            # Regular checkpoint
            if self.checkpoint_dir and step % 500 == 0 and step > 0:
                self.save_checkpoint(f"checkpoint_step_{step}.pt")

        # Final evaluation
        print("\n" + "="*60)
        print("Final Evaluation")
        print("="*60)

        final_metrics = self.evaluate()
        print(f"Final Validation Loss: {final_metrics['val_loss']:.4f}")

        # Save final model
        if self.checkpoint_dir:
            self.save_checkpoint("final_model.pt")
            print(f"✓ Saved final model")

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.model.config.__dict__
        }

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Loaded checkpoint from step {self.step}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")


def create_trainer(
    model,
    train_loader,
    val_loader,
    learning_rate: float = 3e-4,
    num_training_steps: int = 1000,
    warmup_steps: int = 100,
    device: str = 'mps',
    checkpoint_dir: Optional[Path] = None
):
    """
    Create trainer with optimizer and scheduler.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        learning_rate: Learning rate
        num_training_steps: Total training steps
        warmup_steps: Warmup steps
        device: Device to train on
        checkpoint_dir: Checkpoint directory

    Returns:
        Configured trainer
    """
    from .optimizer import configure_optimizer, get_lr_scheduler

    # Create optimizer
    optimizer = configure_optimizer(model, learning_rate=learning_rate)

    # Create scheduler
    scheduler = get_lr_scheduler(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
        scheduler_type='cosine'
    )

    # Create trainer
    trainer = CodeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=checkpoint_dir
    )

    return trainer
