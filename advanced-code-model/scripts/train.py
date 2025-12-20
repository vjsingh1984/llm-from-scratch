"""
Train language model using PyTorch with MPS backend.

Stage 1: Language pretraining on TinyStories
Stage 2: Code fine-tuning on bash scripts

Usage:
    # Stage 1: Language pretraining
    python scripts/train.py --stage language --model-size tiny

    # Stage 2: Code fine-tuning
    python scripts/train.py --stage code --model-size tiny --checkpoint models/language_model.pth
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.config import get_tiny_config, get_config
from model.transformer import create_model as create_dense_model
from model.mamba import create_mamba_model
from model.moe_transformer import create_moe_model
from model.hybrid import create_hybrid_model


def get_device():
    """Get the best available device (MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def load_datasets(stage: str, data_dir: Path, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load training and validation datasets.

    Args:
        stage: 'language' or 'code'
        data_dir: Path to processed data directory
        device: Device (kept on CPU to save memory)

    Returns:
        Train and validation tensors (on CPU)
    """
    if stage == "language":
        train_path = data_dir / "language_train.npy"
        val_path = data_dir / "language_val.npy"
    else:
        train_path = data_dir / "code_train.npy"
        val_path = data_dir / "code_val.npy"

    print(f"Loading {stage} datasets...")
    train_data = np.load(train_path)
    val_data = np.load(val_path)

    print(f"  Train: {train_data.shape}")
    print(f"  Val: {val_data.shape}")

    # Convert to PyTorch tensors - KEEP ON CPU to save memory
    # Data will be moved to device in batches during training
    train_tensor = torch.from_numpy(train_data).long()
    val_tensor = torch.from_numpy(val_data).long()

    return train_tensor, val_tensor


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get a random batch from the dataset.

    Args:
        data: Dataset tensor (num_sequences, seq_len) on CPU
        batch_size: Batch size
        seq_len: Sequence length
        device: Device to move batch to

    Returns:
        Input and target sequences on device
    """
    # Random indices (on CPU)
    num_sequences = data.shape[0]
    indices = torch.randint(0, num_sequences, (batch_size,))

    # Get sequences (on CPU)
    sequences = data[indices]  # (batch_size, seq_len)

    # Input is all tokens except last
    # Target is all tokens except first
    x = sequences[:, :-1]  # (batch_size, seq_len - 1)
    y = sequences[:, 1:]   # (batch_size, seq_len - 1)

    # Move to device only now
    return x.to(device), y.to(device)


def compute_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss.

    Args:
        model: Transformer model
        x: Input tokens (batch_size, seq_len)
        y: Target tokens (batch_size, seq_len)

    Returns:
        Loss value
    """
    # Forward pass
    logits = model(x)  # (batch_size, seq_len, vocab_size)

    # Reshape for loss computation
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
    y_flat = y.reshape(-1)  # (batch_size * seq_len,)

    # Cross-entropy loss (ignore padding tokens with id 0)
    mask = y_flat != 0
    num_tokens = torch.maximum(mask.sum(), torch.tensor(1.0, device=mask.device))

    # Only compute loss on valid (non-padding) tokens
    loss = F.cross_entropy(logits_flat, y_flat, reduction='none')
    masked_loss = torch.where(mask, loss, torch.zeros_like(loss))

    # Average over non-padding tokens
    avg_loss = masked_loss.sum() / num_tokens

    # Clamp loss to prevent numerical issues
    return torch.clamp(avg_loss, 0.0, 20.0)


@torch.no_grad()
def evaluate(model: nn.Module, val_data: torch.Tensor, batch_size: int, device: torch.device, num_batches: int = 10) -> float:
    """
    Evaluate model on validation set.

    Args:
        model: Transformer model
        val_data: Validation dataset (on CPU)
        batch_size: Batch size
        device: Device for computation
        num_batches: Number of batches to evaluate

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    seq_len = val_data.shape[1]

    for _ in range(num_batches):
        x, y = get_batch(val_data, batch_size, seq_len, device)
        loss = compute_loss(model, x, y)
        total_loss += loss.item()

    model.train()
    return total_loss / num_batches


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_data: torch.Tensor,
    batch_size: int,
    num_steps: int,
    device: torch.device,
    epoch: int = 0,
    gradient_accumulation_steps: int = 1,
    scaler: GradScaler = None,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: Transformer model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        train_data: Training dataset (on CPU)
        batch_size: Batch size
        num_steps: Number of training steps
        device: Device for computation
        epoch: Current epoch number

    Returns:
        Training metrics
    """
    model.train()
    total_loss = 0.0
    seq_len = train_data.shape[1]

    # Progress bar
    pbar = tqdm(range(num_steps), desc=f"Epoch {epoch + 1}")

    # For gradient accumulation
    optimizer.zero_grad()

    for step in pbar:
        # Get batch (move to device)
        x, y = get_batch(train_data, batch_size, seq_len, device)

        # Forward pass with optional AMP
        if scaler is not None:
            # Mixed precision training
            with autocast(device_type=device.type):
                loss = compute_loss(model, x, y)
        else:
            # Standard precision
            loss = compute_loss(model, x, y)

        # Scale loss by accumulation steps (so gradient is averaged, not summed)
        loss = loss / gradient_accumulation_steps

        # Check for NaN in loss
        if not torch.isfinite(loss):
            print(f"\n⚠ Warning: NaN/Inf loss at step {step}. Skipping update.")
            continue

        # Backward pass (accumulate gradients)
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Track loss (unscaled for logging)
        loss_val = (loss * gradient_accumulation_steps).item()
        total_loss += loss_val

        # Update weights every N steps
        if (step + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                # Unscale gradients and clip
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Update weights with scaler
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard gradient clipping and update
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

            # Clear MPS cache to prevent memory accumulation
            if device.type == 'mps':
                torch.mps.empty_cache()

        # Update progress bar
        current_lr = scheduler.get_last_lr()[0]
        effective_batch = batch_size * gradient_accumulation_steps
        pbar.set_postfix({
            'loss': f'{loss_val:.4f}',
            'lr': f'{current_lr:.2e}',
            'eff_bs': effective_batch
        })

    avg_loss = total_loss / num_steps
    return {'train_loss': avg_loss}


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, path: Path, metadata: Dict = None):
    """Save model checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save model and optimizer state
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metadata': metadata or {}
    }
    torch.save(checkpoint, path)

    # Save metadata separately for easy reading
    if metadata:
        metadata_path = path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    print(f"✓ Saved checkpoint: {path}")


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, path: Path, device: torch.device):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if 'optimizer_state_dict' in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"✓ Loaded checkpoint: {path}")
    return checkpoint.get('metadata', {})


def main():
    parser = argparse.ArgumentParser(description="Train language model with PyTorch+MPS")
    parser.add_argument("--stage", type=str, required=True, choices=["language", "code"],
                        help="Training stage")
    parser.add_argument("--model-size", type=str, default="tiny",
                        choices=["tiny", "medium", "large", "xlarge"],
                        help="Model size")
    parser.add_argument("--architecture", type=str, default="dense",
                        choices=["dense", "mamba", "moe", "hybrid"],
                        help="Model architecture (dense=Transformer, mamba=SSM, moe=Sparse, hybrid=Mamba+Attention)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size (reduce if OOM)")
    parser.add_argument("--num-epochs", type=int, default=3,
                        help="Number of epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=500,
                        help="Training steps per epoch")
    parser.add_argument("--learning-rate", type=float, default=6e-5,
                        help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100,
                        help="Warmup steps for learning rate")
    parser.add_argument("--eval-interval", type=int, default=500,
                        help="Evaluate every N steps")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Number of steps to accumulate gradients (simulate larger batch size)")
    parser.add_argument("--use-compile", action="store_true",
                        help="Use torch.compile for 20-30% speedup (PyTorch 2.0+)")
    parser.add_argument("--use-rmsnorm", action="store_true",
                        help="Use RMSNorm instead of LayerNorm (10-15% faster normalization)")
    parser.add_argument("--use-amp", action="store_true",
                        help="Use Automatic Mixed Precision for 2x speedup + 30-40%% memory reduction")
    parser.add_argument("--use-gradient-checkpointing", action="store_true",
                        help="Use gradient checkpointing to save 40-50%% memory (trades compute for memory)")
    parser.add_argument("--use-rope", action="store_true",
                        help="Use Rotary Position Embeddings for better length extrapolation (LLaMA-style)")

    # Architecture-specific parameters
    parser.add_argument("--num-experts", type=int, default=8,
                        help="Number of experts for MoE architecture")
    parser.add_argument("--expert-capacity", type=int, default=2,
                        help="Top-K experts to route to (MoE)")
    parser.add_argument("--state-size", type=int, default=16,
                        help="State expansion factor for Mamba/SSM")
    parser.add_argument("--conv-size", type=int, default=4,
                        help="Convolution kernel size for Mamba")
    parser.add_argument("--hybrid-local-window", type=int, default=256,
                        help="Local attention window size for hybrid architecture")

    args = parser.parse_args()

    # Get device
    device = get_device()
    print(f"Using device: {device}")
    print()

    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "processed"
    models_dir = base_dir / "models"

    print("=" * 60)
    print(f"Training Stage: {args.stage.upper()}")
    print("=" * 60)
    print()

    # Load config
    if args.model_size == "tiny":
        config = get_tiny_config()
    else:
        config = get_config(args.model_size)

    # Apply architecture options if requested
    if args.use_rmsnorm:
        config.use_rmsnorm = True
    if args.use_gradient_checkpointing:
        config.use_gradient_checkpointing = True
    if args.use_rope:
        config.use_rope = True

    # Apply architecture-specific parameters
    config.num_experts = args.num_experts
    config.expert_capacity = args.expert_capacity
    config.state_size = args.state_size
    config.conv_size = args.conv_size
    config.hybrid_local_window = args.hybrid_local_window

    print(f"Model: {args.model_size.upper()} ({args.architecture.upper()})")
    print(f"  Parameters: {config.num_parameters}M (approx)")
    print(f"  Layers: {config.n_layers}")
    print(f"  Hidden dim: {config.d_model}")
    print(f"  Attention heads: {config.n_heads}")
    print(f"  FFN dim: {config.d_ff}")
    print(f"  Max sequence: {config.max_seq_len}")
    print(f"  Vocabulary: {config.vocab_size}")
    if args.architecture == "moe":
        print(f"  Experts: {config.num_experts}")
        print(f"  Expert capacity: {config.expert_capacity}")
    elif args.architecture == "mamba":
        print(f"  State size: {config.state_size}")
        print(f"  Conv size: {config.conv_size}")
    elif args.architecture == "hybrid":
        print(f"  State size: {config.state_size}")
        print(f"  Local window: {config.hybrid_local_window}")
    print()

    # Create model based on architecture
    print(f"Creating model ({args.architecture})...")
    if args.architecture == "dense":
        model = create_dense_model(config, device=str(device))
    elif args.architecture == "mamba":
        model = create_mamba_model(config, device=str(device))
    elif args.architecture == "moe":
        model = create_moe_model(config, device=str(device))
    elif args.architecture == "hybrid":
        model = create_hybrid_model(config, device=str(device))
    else:
        raise ValueError(f"Unknown architecture: {args.architecture}")
    print(f"✓ Model created")
    print()

    # Load datasets
    train_data, val_data = load_datasets(args.stage, data_dir, device)
    print()

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )

    # Create learning rate scheduler with warmup
    total_steps = args.num_epochs * args.steps_per_epoch

    def lr_lambda(current_step):
        if current_step < args.warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, args.warmup_steps))
        else:
            # Cosine decay after warmup
            progress = float(current_step - args.warmup_steps) / float(max(1, total_steps - args.warmup_steps))
            return max(0.1, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Load checkpoint if provided (BEFORE torch.compile!)
    if args.checkpoint:
        load_checkpoint(model, optimizer, Path(args.checkpoint), device)
        print()

    # Apply torch.compile AFTER loading checkpoint (PyTorch 2.0+)
    if args.use_compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode='default')
        print("✓ Model compiled (expect 20-30% speedup after warmup)")
        print()

    # Create GradScaler for mixed precision training
    scaler = None
    if args.use_amp:
        if device.type == 'mps':
            print("⚠ Warning: AMP has limited support on MPS. Using it anyway but may not see full benefits.")
        scaler = GradScaler(device.type)

    # Training info
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print("Training Configuration:")
    print(f"  Stage: {args.stage}")
    print(f"  Device: {device}")
    print(f"  Batch size: {args.batch_size}")
    if args.gradient_accumulation_steps > 1:
        print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Steps/epoch: {args.steps_per_epoch}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Total steps: {total_steps}")
    if args.use_compile:
        print(f"  torch.compile: enabled")
    if args.use_rmsnorm:
        print(f"  RMSNorm: enabled (faster normalization)")
    if args.use_amp:
        print(f"  Mixed Precision (AMP): enabled (2x speedup + memory savings)")
    if args.use_gradient_checkpointing:
        print(f"  Gradient Checkpointing: enabled (40-50% less memory, 20% slower)")
    if args.use_rope:
        print(f"  RoPE: enabled (better position encoding)")
    print()

    # Training loop
    print("=" * 60)
    print("Training")
    print("=" * 60)
    print()

    best_val_loss = float('inf')
    training_history = []

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print("-" * 60)

        # Train
        start_time = time.time()
        metrics = train_epoch(
            model, optimizer, scheduler, train_data,
            args.batch_size, args.steps_per_epoch, device, epoch,
            args.gradient_accumulation_steps, scaler
        )
        epoch_time = time.time() - start_time

        # Evaluate
        print("\nEvaluating...")
        val_loss = evaluate(model, val_data, args.batch_size, device, num_batches=20)

        # Print metrics
        print(f"Train loss: {metrics['train_loss']:.4f}")
        print(f"Val loss: {val_loss:.4f}")
        print(f"Time: {epoch_time:.1f}s")
        print()

        # Save metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': metrics['train_loss'],
            'val_loss': val_loss,
            'time': epoch_time,
        }
        training_history.append(epoch_metrics)

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = models_dir / f"{args.stage}_model_best.pth"
            metadata = {
                'stage': args.stage,
                'model_size': args.model_size,
                'epoch': epoch + 1,
                'val_loss': val_loss,
                'config': {
                    'vocab_size': config.vocab_size,
                    'd_model': config.d_model,
                    'n_layers': config.n_layers,
                    'n_heads': config.n_heads,
                    'd_ff': config.d_ff,
                    'max_seq_len': config.max_seq_len,
                }
            }
            save_checkpoint(model, optimizer, checkpoint_path, metadata)

        # Save latest checkpoint
        latest_path = models_dir / f"{args.stage}_model_latest.pth"
        save_checkpoint(model, optimizer, latest_path)

        print()

    # Save training history
    history_path = models_dir / f"{args.stage}_training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)

    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print()
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {models_dir}")
    print()

    if args.stage == "language":
        print("Next step: Fine-tune on code")
        print(f"  python scripts/train.py --stage code --checkpoint {models_dir}/language_model_best.pth")


if __name__ == "__main__":
    main()
