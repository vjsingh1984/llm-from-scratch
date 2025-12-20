"""
Train bash code generation model.

End-to-end training pipeline:
1. Load tokenizer
2. Load data
3. Create model
4. Train
5. Save model
"""

import sys
from pathlib import Path
import torch
import argparse

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer import CodeTokenizer
from model import create_model
from training import create_dataloaders, create_trainer


def main():
    parser = argparse.ArgumentParser(description='Train bash code model')

    # Model
    parser.add_argument('--model-size', type=str, default='tiny',
                        choices=['tiny', 'small', 'medium'],
                        help='Model size')

    # Data
    parser.add_argument('--data-dir', type=Path,
                        default=Path(__file__).parent.parent / 'data',
                        help='Data directory')
    parser.add_argument('--max-seq-len', type=int, default=256,
                        help='Maximum sequence length')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')

    # Training
    parser.add_argument('--num-steps', type=int, default=1000,
                        help='Number of training steps')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--warmup-steps', type=int, default=100,
                        help='Warmup steps')

    # Device
    parser.add_argument('--device', type=str, default=None,
                        choices=['mps', 'cuda', 'cpu'],
                        help='Device to train on (default: auto-detect)')

    # Output
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent.parent / 'checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--tokenizer-dir', type=Path,
                        default=Path(__file__).parent.parent / 'tokenizer_trained',
                        help='Directory to save trained tokenizer')

    args = parser.parse_args()

    print("="*60)
    print("Bash Code Model Training")
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
    print(f"  Model size: {args.model_size}")
    print(f"  Device: {args.device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max seq len: {args.max_seq_len}")
    print(f"  Training steps: {args.num_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print()

    # Step 1: Create tokenizer
    print("="*60)
    print("Step 1: Creating Tokenizer")
    print("="*60)

    tokenizer = CodeTokenizer()
    tokenizer.build_default_vocab()

    print(f"Vocabulary size: {len(tokenizer)}")

    # Save tokenizer
    tokenizer.save(args.tokenizer_dir)

    # Step 2: Load data
    print("\n" + "="*60)
    print("Step 2: Loading Data")
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

    # Step 3: Create model
    print("\n" + "="*60)
    print("Step 3: Creating Model")
    print("="*60)

    model = create_model(
        model_size=args.model_size,
        vocab_size=len(tokenizer),
        device=args.device
    )

    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Device: {model.get_device()}")

    # Step 4: Create trainer
    print("\n" + "="*60)
    print("Step 4: Setting up Training")
    print("="*60)

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

    # Step 5: Train
    print("\n" + "="*60)
    print("Step 5: Training")
    print("="*60)

    trainer.train(
        num_steps=args.num_steps,
        tokenizer=tokenizer,
        log_interval=10
    )

    # Step 6: Summary
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nCheckpoints saved to: {args.output_dir}")
    print(f"Tokenizer saved to: {args.tokenizer_dir}")
    print("\nTo generate code:")
    print(f"  python scripts/generate.py --checkpoint {args.output_dir}/best_model.pt")


if __name__ == '__main__':
    main()
