"""
Train a bilingual model: English (from TinyStories) + Bash Code

This demonstrates the modern approach to code models:
1. Start with language-pretrained model
2. Continue training on code
3. Result: Model that understands English prompts AND generates code

Uses our existing LLM trained on TinyStories!
"""

import sys
from pathlib import Path
import torch
import argparse
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer import CodeTokenizer
from model import CodeTransformer
from model.config import CoderConfig
from training import create_dataloaders


def create_combined_dataset(
    language_data_path: Path,
    code_data_path: Path,
    output_path: Path,
    language_ratio: float = 0.7
):
    """
    Create combined language + code dataset.

    Args:
        language_data_path: Path to language data (TinyStories)
        code_data_path: Path to code data (bash scripts)
        output_path: Where to save combined dataset
        language_ratio: Ratio of language to code (0.7 = 70% language, 30% code)
    """
    print("\nCreating Combined Dataset...")
    print("="*60)

    # Load language data
    language_texts = []

    # Try TinyStories format
    stories_file = language_data_path / "tiny_stories.txt"
    if stories_file.exists():
        with open(stories_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Split on double newlines (story separator)
            language_texts = [s.strip() for s in content.split('\n\n') if s.strip()]

    print(f"  Language samples: {len(language_texts)}")

    # Load code data
    code_file = code_data_path / "bash_scripts.json"
    with open(code_file, 'r') as f:
        code_data = json.load(f)
        code_texts = code_data['scripts']

    print(f"  Code samples: {len(code_texts)}")

    # Calculate how many code samples we need to match ratio
    target_code_count = int(len(language_texts) * (1 - language_ratio) / language_ratio)

    # Repeat code samples if needed
    code_repeat = max(1, target_code_count // len(code_texts))
    code_texts_repeated = code_texts * code_repeat

    print(f"  Code samples (repeated): {len(code_texts_repeated)}")

    # Combine
    combined_texts = language_texts + code_texts_repeated

    # Shuffle
    import random
    random.seed(42)  # Reproducible
    random.shuffle(combined_texts)

    # Save
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "combined.json", 'w') as f:
        json.dump({
            'texts': combined_texts,
            'count': len(combined_texts),
            'language_count': len(language_texts),
            'code_count': len(code_texts_repeated),
            'language_ratio': len(language_texts) / len(combined_texts),
            'code_ratio': len(code_texts_repeated) / len(combined_texts)
        }, f, indent=2)

    # Also save as plain text for verification
    with open(output_path / "combined.txt", 'w') as f:
        for text in combined_texts:
            f.write(text)
            f.write("\n\n" + "="*60 + "\n\n")

    print(f"\nCombined dataset created:")
    print(f"  Total samples: {len(combined_texts)}")
    print(f"  Language: {len(language_texts)} ({len(language_texts)/len(combined_texts):.1%})")
    print(f"  Code: {len(code_texts_repeated)} ({len(code_texts_repeated)/len(combined_texts):.1%})")
    print(f"  Saved to: {output_path}")

    return output_path


def adapt_tokenizer_for_code(tokenizer, code_samples):
    """
    Ensure tokenizer can handle both English and code.

    Character-level tokenizer already handles both, but this
    function demonstrates how you might extend vocabulary.
    """
    # Count characters in code that might not be in vocabulary
    missing_chars = set()

    for script in code_samples:
        for char in script:
            if char not in tokenizer.vocab:
                missing_chars.add(char)

    if missing_chars:
        print(f"\n  Warning: {len(missing_chars)} characters not in vocabulary")
        print(f"  Missing: {sorted(missing_chars)}")

        # Add missing characters
        next_id = len(tokenizer.vocab)
        for char in sorted(missing_chars):
            tokenizer.vocab[char] = next_id
            tokenizer.id_to_token[next_id] = char
            next_id += 1

        print(f"  Extended vocabulary to {len(tokenizer.vocab)} tokens")

    return tokenizer


def main():
    parser = argparse.ArgumentParser(
        description='Train bilingual model (Language + Code)'
    )

    # Data paths
    parser.add_argument('--language-data', type=Path,
                        default=Path(__file__).parent.parent.parent / 'llm-from-scratch' / 'data',
                        help='Path to language data (TinyStories)')
    parser.add_argument('--code-data', type=Path,
                        default=Path(__file__).parent.parent / 'data_large',
                        help='Path to code data (bash scripts)')
    parser.add_argument('--combined-data', type=Path,
                        default=Path(__file__).parent.parent / 'data_combined',
                        help='Where to save combined dataset')

    # Model
    parser.add_argument('--model-size', type=str, default='tiny',
                        choices=['tiny', 'small', 'medium'],
                        help='Model size')
    parser.add_argument('--vocab-size', type=int, default=256,
                        help='Vocabulary size (256 for extended char-level)')

    # Training
    parser.add_argument('--num-steps', type=int, default=2000,
                        help='Total training steps')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--warmup-steps', type=int, default=200,
                        help='Warmup steps')

    # Dataset mixing
    parser.add_argument('--language-ratio', type=float, default=0.7,
                        help='Ratio of language to code (0.7 = 70% language)')

    # Device
    parser.add_argument('--device', type=str, default=None,
                        choices=['mps', 'cuda', 'cpu'])

    # Output
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent.parent / 'checkpoints_bilingual',
                        help='Output directory')

    args = parser.parse_args()

    print("="*60)
    print("Bilingual Model Training")
    print("Language (English) + Code (Bash)")
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
    print(f"  Language ratio: {args.language_ratio:.0%}")
    print(f"  Code ratio: {1-args.language_ratio:.0%}")
    print(f"  Device: {args.device}")
    print(f"  Training steps: {args.num_steps}")
    print()

    # Step 1: Create combined dataset
    print("="*60)
    print("Step 1: Preparing Combined Dataset")
    print("="*60)

    combined_path = create_combined_dataset(
        language_data_path=args.language_data,
        code_data_path=args.code_data,
        output_path=args.combined_data,
        language_ratio=args.language_ratio
    )

    # Step 2: Create tokenizer
    print("\n" + "="*60)
    print("Step 2: Creating Bilingual Tokenizer")
    print("="*60)

    tokenizer = CodeTokenizer()
    tokenizer.build_default_vocab()

    # Extend for any code-specific characters
    code_file = args.code_data / "bash_scripts.json"
    with open(code_file) as f:
        code_data = json.load(f)
        tokenizer = adapt_tokenizer_for_code(tokenizer, code_data['scripts'])

    print(f"\nFinal vocabulary size: {len(tokenizer)}")

    # Save tokenizer
    tokenizer_dir = Path(__file__).parent.parent / 'tokenizer_bilingual'
    tokenizer.save(tokenizer_dir)

    # Step 3: Load data
    print("\n" + "="*60)
    print("Step 3: Loading Training Data")
    print("="*60)

    from training import create_dataloaders

    train_loader, val_loader = create_dataloaders(
        data_path=combined_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=256,
        train_split=0.9
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Step 4: Create model
    print("\n" + "="*60)
    print("Step 4: Creating Model")
    print("="*60)

    from model import create_model

    model = create_model(
        model_size=args.model_size,
        vocab_size=len(tokenizer),
        device=args.device
    )

    # Step 5: Train
    print("\n" + "="*60)
    print("Step 5: Training Bilingual Model")
    print("="*60)
    print("This model will learn BOTH:")
    print("  1. English language patterns (from TinyStories)")
    print("  2. Bash code patterns (from scripts)")
    print()

    from training import create_trainer

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
        log_interval=20
    )

    # Step 6: Test both capabilities
    print("\n" + "="*60)
    print("Step 6: Testing Bilingual Capabilities")
    print("="*60)

    print("\nTest 1: English Generation")
    print("-"*60)
    english_prompt = "Once upon a time"
    sample = trainer.generate_sample(english_prompt, tokenizer, max_tokens=50)
    print(f"Prompt: {repr(english_prompt)}")
    print(f"Generated:\n{sample}")

    print("\nTest 2: Code Generation")
    print("-"*60)
    code_prompt = "#!/bin/bash\n"
    sample = trainer.generate_sample(code_prompt, tokenizer, max_tokens=50)
    print(f"Prompt: {repr(code_prompt)}")
    print(f"Generated:\n{sample}")

    # Summary
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nModel saved to: {args.output_dir}")
    print(f"Tokenizer saved to: {tokenizer_dir}")

    print("\nYour model can now:")
    print("  ✓ Understand English prompts")
    print("  ✓ Generate bash code")
    print("  ✓ Mix language and code")

    print("\nNext steps:")
    print("  1. Test with English prompts like 'write a backup script'")
    print("  2. Fine-tune further on more code")
    print("  3. Build a chat interface")

    print(f"\nGenerate code:")
    print(f"  python scripts/generate.py \\")
    print(f"    --checkpoint {args.output_dir}/best_model.pt \\")
    print(f"    --tokenizer-dir {tokenizer_dir} \\")
    print(f'    --prompt "Create a script to backup files"')


if __name__ == '__main__':
    main()
