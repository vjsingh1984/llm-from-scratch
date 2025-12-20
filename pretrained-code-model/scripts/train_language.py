"""
Stage 1: Pretrain language model on natural language (TinyStories).

This creates the foundation - model learns:
- English vocabulary and grammar
- Reasoning and logic
- Instruction following
- Common knowledge

Then we'll fine-tune on code in Stage 2.
"""

import sys
from pathlib import Path
import torch
import json
import argparse

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer import BPETokenizer
from model import create_model
from training import create_dataloaders, create_trainer


def train_bpe_tokenizer(texts, vocab_size=8000):
    """Train BPE tokenizer on texts."""
    print("="*60)
    print("Training BPE Tokenizer")
    print("="*60)
    print()

    tokenizer = BPETokenizer()

    # Set target vocab size
    tokenizer.target_vocab_size = vocab_size

    # Sample for training (use subset if too large)
    sample_size = min(10000, len(texts))
    sample_texts = texts[:sample_size]

    print(f"Training on {sample_size} samples...")
    print(f"Target vocab size: {vocab_size}")
    tokenizer.train(sample_texts, verbose=True)

    print(f"\nTokenizer trained: {len(tokenizer.vocab)} tokens")

    return tokenizer


def main():
    parser = argparse.ArgumentParser(description='Stage 1: Language Pretraining')

    # Data
    parser.add_argument('--data-dir', type=Path,
                        default=Path(__file__).parent.parent / 'data_language',
                        help='Language data directory')

    # Tokenizer
    parser.add_argument('--vocab-size', type=int, default=8000,
                        help='BPE vocabulary size')
    parser.add_argument('--tokenizer-dir', type=Path,
                        default=Path(__file__).parent.parent / 'tokenizer_trained',
                        help='Where to save tokenizer')

    # Model
    parser.add_argument('--model-size', type=str, default='small',
                        choices=['tiny', 'small', 'medium'],
                        help='Model size')

    # Training
    parser.add_argument('--num-steps', type=int, default=5000,
                        help='Training steps')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--max-seq-len', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--warmup-steps', type=int, default=500,
                        help='Warmup steps')

    # Device
    parser.add_argument('--device', type=str, default=None,
                        choices=['mps', 'cuda', 'cpu'])

    # Output
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent.parent / 'checkpoints_language',
                        help='Checkpoint output directory')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("STAGE 1: Language Pretraining")
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
    print(f"  Model: {args.model_size}")
    print(f"  Device: {args.device}")
    print(f"  Steps: {args.num_steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print()

    # Step 1: Load language data
    print("="*60)
    print("Step 1: Loading Language Data")
    print("="*60)

    data_file = args.data_dir / "language_data.json"

    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        print("\nRun first:")
        print("  python scripts/download_language_data.py")
        return

    with open(data_file) as f:
        data = json.load(f)
        texts = data['texts']

    print(f"Loaded {len(texts):,} texts")

    # Step 2: Train tokenizer
    print("\n" + "="*60)
    print("Step 2: Training Tokenizer")
    print("="*60)

    tokenizer = train_bpe_tokenizer(texts, vocab_size=args.vocab_size)

    # Save tokenizer
    tokenizer.save(args.tokenizer_dir)

    # Step 3: Prepare training data
    print("\n" + "="*60)
    print("Step 3: Preparing Training Data")
    print("="*60)

    # For BPE tokenizer with text data, we need to adapt the dataloader
    # Save texts in the format expected by dataloader
    temp_data_dir = args.data_dir / "processed"
    temp_data_dir.mkdir(exist_ok=True)

    # Save as JSON in the format create_dataloaders expects
    with open(temp_data_dir / "language_data.json", 'w') as f:
        # create_dataloaders expects {'scripts': [...]}
        json.dump({'scripts': texts}, f, indent=2)

    train_loader, val_loader = create_dataloaders(
        data_path=temp_data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_seq_len,
        train_split=0.9
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Step 4: Create model
    print("\n" + "="*60)
    print("Step 4: Creating Language Model")
    print("="*60)

    model = create_model(
        model_size=args.model_size,
        vocab_size=len(tokenizer.vocab),
        device=args.device
    )

    print(f"  Parameters: {model.count_parameters():,}")

    # Step 5: Train
    print("\n" + "="*60)
    print("Step 5: Training on Language Data")
    print("="*60)
    print()
    print("The model is learning:")
    print("  ✓ English vocabulary and grammar")
    print("  ✓ Reasoning and logic")
    print("  ✓ Instruction following")
    print()

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

    trainer.train(
        num_steps=args.num_steps,
        tokenizer=tokenizer,
        log_interval=50
    )

    # Step 6: Test language generation
    print("\n" + "="*60)
    print("Step 6: Testing Language Model")
    print("="*60)

    test_prompts = [
        "Once upon a time",
        "The little girl",
        "In a small village"
    ]

    for prompt in test_prompts:
        generated = trainer.generate_sample(prompt, tokenizer, max_tokens=50)
        print(f"\nPrompt: {repr(prompt)}")
        print(f"Generated:\n{generated[:200]}")

    # Summary
    print("\n" + "="*60)
    print("Stage 1 Complete: Language Model Trained!")
    print("="*60)
    print(f"\nModel saved to: {args.output_dir}")
    print(f"Tokenizer saved to: {args.tokenizer_dir}")

    print("\nNext: Stage 2 - Fine-tune on code")
    print("  python scripts/train_code.py \\")
    print(f"    --language-checkpoint {args.output_dir}/best_model.pt")


if __name__ == '__main__':
    main()
