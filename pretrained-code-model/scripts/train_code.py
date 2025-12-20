"""
Stage 2: Fine-tune language model on bash code.

Takes the pretrained language model and continues training on code.
Model retains English understanding while learning to generate code.

This is how CodeLlama, StarCoder, and GitHub Copilot are built!
"""

import sys
from pathlib import Path
import torch
import json
import argparse

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer import BPETokenizer
from model import CodeTransformer
from model.config import CoderConfig
from training import create_dataloaders, create_trainer


def load_language_model(checkpoint_path: Path, device: str):
    """Load the pretrained language model."""
    print("="*60)
    print("Loading Pretrained Language Model")
    print("="*60)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Recreate config
    config = CoderConfig(**checkpoint['config'])

    # Create and load model
    model = CodeTransformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"✓ Model loaded")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Previous training: {checkpoint['step']} steps")
    print(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")

    return model, config


def main():
    parser = argparse.ArgumentParser(description='Stage 2: Code Fine-tuning')

    # Model
    parser.add_argument('--language-checkpoint', type=Path,
                        default=Path(__file__).parent.parent / 'checkpoints_language' / 'best_model.pt',
                        help='Language model checkpoint')
    parser.add_argument('--tokenizer-dir', type=Path,
                        default=Path(__file__).parent.parent / 'tokenizer_trained',
                        help='Tokenizer directory')

    # Data
    parser.add_argument('--data-dir', type=Path,
                        default=Path(__file__).parent.parent / 'data_code',
                        help='Code data directory')

    # Training
    parser.add_argument('--num-steps', type=int, default=2000,
                        help='Fine-tuning steps')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--max-seq-len', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate (lower for fine-tuning)')
    parser.add_argument('--warmup-steps', type=int, default=100,
                        help='Warmup steps')

    # Device
    parser.add_argument('--device', type=str, default=None,
                        choices=['mps', 'cuda', 'cpu'])

    # Output
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent.parent / 'checkpoints_code',
                        help='Checkpoint output directory')
    parser.add_argument('--final-dir', type=Path,
                        default=Path(__file__).parent.parent / 'checkpoints_final',
                        help='Final bilingual model directory')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("STAGE 2: Code Fine-tuning")
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
    print(f"  Device: {args.device}")
    print(f"  Steps: {args.num_steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate} (fine-tuning)")
    print()

    # Step 1: Load tokenizer
    print("="*60)
    print("Step 1: Loading Tokenizer")
    print("="*60)

    tokenizer = BPETokenizer.load(args.tokenizer_dir)
    print(f"Vocabulary size: {len(tokenizer.vocab)}")

    # Step 2: Load pretrained model
    print("\n" + "="*60)
    print("Step 2: Loading Pretrained Model")
    print("="*60)

    if not args.language_checkpoint.exists():
        print(f"Error: Language checkpoint not found: {args.language_checkpoint}")
        print("\nRun Stage 1 first:")
        print("  python scripts/train_language.py")
        return

    model, config = load_language_model(args.language_checkpoint, args.device)

    # Step 3: Load code data
    print("\n" + "="*60)
    print("Step 3: Loading Code Data")
    print("="*60)

    data_file = args.data_dir / "code_data.json"

    if not data_file.exists():
        print(f"Error: Code data not found: {data_file}")
        print("\nRun download script:")
        print("  python scripts/download_code_data.py")
        return

    with open(data_file) as f:
        data = json.load(f)

    print(f"Loaded {data['count']:,} bash scripts")

    # Prepare for dataloader
    temp_data_dir = args.data_dir / "processed"
    temp_data_dir.mkdir(exist_ok=True)

    # Save in expected format
    with open(temp_data_dir / "code_data.json", 'w') as f:
        json.dump({'scripts': data['scripts']}, f, indent=2)

    train_loader, val_loader = create_dataloaders(
        data_path=temp_data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_seq_len,
        train_split=0.9
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Step 4: Fine-tune on code
    print("\n" + "="*60)
    print("Step 4: Fine-tuning on Bash Code")
    print("="*60)
    print()
    print("The model is learning:")
    print("  ✓ Bash syntax and commands")
    print("  ✓ Code patterns and idioms")
    print("  ✓ Script structure")
    print()
    print("While retaining:")
    print("  ✓ English understanding")
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

    # Step 5: Test bilingual capabilities
    print("\n" + "="*60)
    print("Step 5: Testing Bilingual Model")
    print("="*60)

    print("\nTest 1: English generation")
    print("-"*60)
    english_prompts = ["Once upon a time", "The little girl"]

    for prompt in english_prompts:
        generated = trainer.generate_sample(prompt, tokenizer, max_tokens=40)
        print(f"\nPrompt: {repr(prompt)}")
        print(f"Generated:\n{generated[:150]}")

    print("\n\nTest 2: Code generation")
    print("-"*60)
    code_prompts = ["#!/bin/bash\n", "for i in"]

    for prompt in code_prompts:
        generated = trainer.generate_sample(prompt, tokenizer, max_tokens=40)
        print(f"\nPrompt: {repr(prompt)}")
        print(f"Generated:\n{generated[:150]}")

    # Step 6: Copy best model to final directory
    print("\n" + "="*60)
    print("Step 6: Saving Final Model")
    print("="*60)

    import shutil

    args.final_dir.mkdir(parents=True, exist_ok=True)

    best_model = args.output_dir / "best_model.pt"
    if best_model.exists():
        shutil.copy(best_model, args.final_dir / "best_model.pt")
        print(f"✓ Copied to {args.final_dir}/best_model.pt")

    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print()
    print("You now have a bilingual model that:")
    print("  ✓ Understands English prompts")
    print("  ✓ Generates bash code")
    print("  ✓ Can respond in both language and code")
    print()
    print(f"Model saved to: {args.final_dir}/best_model.pt")
    print(f"Tokenizer: {args.tokenizer_dir}")
    print()
    print("Test it:")
    print("  python scripts/generate.py \\")
    print(f"    --checkpoint {args.final_dir}/best_model.pt \\")
    print('    --prompt "Create a backup script"')


if __name__ == '__main__':
    main()
