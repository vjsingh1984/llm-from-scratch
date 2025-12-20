"""
Continue training from a checkpoint with new/additional data.

This demonstrates:
1. Loading a pre-trained model
2. Fine-tuning on new data
3. Incremental learning
"""

import sys
from pathlib import Path
import torch
import argparse

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer import CodeTokenizer
from model import CodeTransformer
from model.config import CoderConfig
from training import create_dataloaders, create_trainer


def load_checkpoint(checkpoint_path: Path, device: str):
    """
    Load model from checkpoint.

    Returns:
        Tuple of (model, config, training_state)
    """
    print(f"Loading checkpoint from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Recreate config
    config = CoderConfig(**checkpoint['config'])

    # Create and load model
    model = CodeTransformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"  Model loaded successfully")
    print(f"  Previous training step: {checkpoint['step']}")
    print(f"  Previous best val loss: {checkpoint['best_val_loss']:.4f}")
    print(f"  Parameters: {model.count_parameters():,}")

    training_state = {
        'step': checkpoint['step'],
        'best_val_loss': checkpoint['best_val_loss']
    }

    return model, config, training_state


def main():
    parser = argparse.ArgumentParser(description='Continue training from checkpoint')

    # Checkpoint
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Path to checkpoint to continue from')
    parser.add_argument('--tokenizer-dir', type=Path,
                        default=Path(__file__).parent.parent / 'tokenizer_trained',
                        help='Tokenizer directory')

    # Data
    parser.add_argument('--data-dir', type=Path,
                        default=Path(__file__).parent.parent / 'data_large',
                        help='Data directory (can be different from original)')
    parser.add_argument('--max-seq-len', type=int, default=256,
                        help='Maximum sequence length')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')

    # Training
    parser.add_argument('--num-steps', type=int, default=1000,
                        help='Number of ADDITIONAL training steps')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate (usually lower for fine-tuning)')
    parser.add_argument('--warmup-steps', type=int, default=50,
                        help='Warmup steps')

    # Device
    parser.add_argument('--device', type=str, default=None,
                        choices=['mps', 'cuda', 'cpu'],
                        help='Device to train on (default: auto-detect)')

    # Output
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent.parent / 'checkpoints_continued',
                        help='Output directory for new checkpoints')

    args = parser.parse_args()

    print("="*60)
    print("Continue Training (Fine-tuning)")
    print("="*60)
    print()

    # Auto-detect device
    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = 'mps'
        elif torch.cuda.is_available():
            args.device = 'cuda'
        else:
            args.device = 'cpu'

    print(f"Configuration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Device: {args.device}")
    print(f"  Additional steps: {args.num_steps}")
    print(f"  Learning rate: {args.learning_rate} (fine-tuning)")
    print()

    # Step 1: Load tokenizer
    print("="*60)
    print("Step 1: Loading Tokenizer")
    print("="*60)

    tokenizer = CodeTokenizer.load(args.tokenizer_dir)
    print(f"Vocabulary size: {len(tokenizer)}")

    # Step 2: Load model from checkpoint
    print("\n" + "="*60)
    print("Step 2: Loading Pre-trained Model")
    print("="*60)

    model, config, training_state = load_checkpoint(args.checkpoint, args.device)

    # Step 3: Load new data
    print("\n" + "="*60)
    print("Step 3: Loading Training Data")
    print("="*60)

    train_loader, val_loader = create_dataloaders(
        data_path=args.data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_seq_len,
        train_split=0.9
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Step 4: Create trainer for fine-tuning
    print("\n" + "="*60)
    print("Step 4: Setting up Fine-tuning")
    print("="*60)

    # Use lower learning rate for fine-tuning (important!)
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        num_training_steps=args.num_steps,
        warmup_steps=args.warmup_steps,
        device=args.device,
        checkpoint_dir=args.output_dir
    )

    # Update trainer state
    trainer.step = training_state['step']
    trainer.best_val_loss = training_state['best_val_loss']

    print(f"  Starting from step: {trainer.step}")
    print(f"  Target step: {trainer.step + args.num_steps}")

    # Step 5: Continue training
    print("\n" + "="*60)
    print("Step 5: Fine-tuning Model")
    print("="*60)
    print("Note: This continues training from the previous checkpoint")
    print()

    trainer.train(
        num_steps=args.num_steps,
        tokenizer=tokenizer,
        log_interval=10
    )

    # Step 6: Summary
    print("\n" + "="*60)
    print("Fine-tuning Complete!")
    print("="*60)
    print(f"\nNew checkpoints saved to: {args.output_dir}")
    print(f"\nComparison:")
    print(f"  Original best val loss: {training_state['best_val_loss']:.4f}")
    print(f"  New best val loss: {trainer.best_val_loss:.4f}")

    if trainer.best_val_loss < training_state['best_val_loss']:
        improvement = training_state['best_val_loss'] - trainer.best_val_loss
        print(f"  ✓ Improved by {improvement:.4f}")
    else:
        print(f"  Model did not improve (may need more data or different LR)")

    print("\nTo generate code with fine-tuned model:")
    print(f"  python scripts/generate.py --checkpoint {args.output_dir}/best_model.pt")


if __name__ == '__main__':
    main()
