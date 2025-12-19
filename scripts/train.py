"""
Training script for LLM.

Example usage:
    python scripts/train.py --model-size tiny --vocab-size 8000
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
from model import GPTModel, GPTConfig, create_model
from training import Trainer, TrainerConfig, create_train_val_loaders
from tokenizer import BPETokenizer
import argparse


def main():
    parser = argparse.ArgumentParser(description='Train a language model')

    # Model configuration
    parser.add_argument('--model-size', type=str, default='tiny',
                       choices=['tiny', 'gpt2-small', 'gpt2-medium'],
                       help='Model size')
    parser.add_argument('--vocab-size', type=int, default=8000,
                       help='Vocabulary size')
    parser.add_argument('--seq-len', type=int, default=256,
                       help='Sequence length')
    parser.add_argument('--n-layers', type=int, default=None,
                       help='Number of layers (overrides model-size)')
    parser.add_argument('--d-model', type=int, default=None,
                       help='Model dimension (overrides model-size)')

    # Data
    parser.add_argument('--train-data', type=str,
                       default='data/tinystories_10000.txt',
                       help='Path to training data')
    parser.add_argument('--val-data', type=str,
                       default='data/tinystories_validation.txt',
                       help='Path to validation data')
    parser.add_argument('--tokenizer-path', type=str,
                       default='tokenizer_model',
                       help='Path to trained tokenizer')

    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--max-epochs', type=int, default=10,
                       help='Maximum epochs')
    parser.add_argument('--max-steps', type=int, default=None,
                       help='Maximum steps (overrides max-epochs)')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.1,
                       help='Weight decay')
    parser.add_argument('--warmup-steps', type=int, default=500,
                       help='Warmup steps')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                       help='Gradient clipping')

    # Logging and checkpointing
    parser.add_argument('--log-interval', type=int, default=10,
                       help='Log every N steps')
    parser.add_argument('--eval-interval', type=int, default=500,
                       help='Evaluate every N steps')
    parser.add_argument('--save-interval', type=int, default=1000,
                       help='Save checkpoint every N steps')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Checkpoint directory')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')

    args = parser.parse_args()

    # Print configuration
    print("="*60)
    print("Training Configuration")
    print("="*60)
    print(f"Model size: {args.model_size}")
    print(f"Vocabulary size: {args.vocab_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("="*60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer_path = Path(args.tokenizer_path)
    if not tokenizer_path.exists():
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        print("Please train a tokenizer first using:")
        print("  python scripts/download_data.py")
        print("  python scripts/train_tokenizer.py")
        sys.exit(1)

    tokenizer = BPETokenizer.load(tokenizer_path)
    print(f"Loaded tokenizer with vocab size: {len(tokenizer.vocab)}")

    # Create model
    print("\nCreating model...")
    kwargs = {}
    if args.n_layers is not None:
        kwargs['n_layers'] = args.n_layers
    if args.d_model is not None:
        kwargs['d_model'] = args.d_model

    model = create_model(
        model_size=args.model_size,
        vocab_size=len(tokenizer.vocab),
        max_seq_len=args.seq_len,
        **kwargs
    )

    n_params = model.count_parameters()
    print(f"Model created with {n_params:,} parameters ({n_params/1e6:.2f}M)")

    # Load data
    print("\nLoading data...")
    train_file = Path(args.train_data)
    val_file = Path(args.val_data)

    if not train_file.exists():
        print(f"Error: Training data not found at {train_file}")
        print("Please download data first using:")
        print("  python scripts/download_data.py")
        sys.exit(1)

    train_loader, val_loader = create_train_val_loaders(
        train_file=train_file,
        val_file=val_file if val_file.exists() else None,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size
    )

    print(f"Training batches: {len(train_loader)}")
    if val_loader:
        print(f"Validation batches: {len(val_loader)}")

    # Create trainer config
    trainer_config = TrainerConfig(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_clip=args.gradient_clip,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        config=trainer_config,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer
    )

    # Resume if requested
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print("\nStarting training...\n")
    trainer.train()

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"Checkpoints saved to: {args.checkpoint_dir}")


if __name__ == '__main__':
    main()
