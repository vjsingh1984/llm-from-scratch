#!/usr/bin/env python3
"""
Train a BPE tokenizer optimized for code.

Trains on code corpus with proper handling of:
- Bash scripts, shebangs
- Python code (def, class, imports)
- Special characters ({}, [], (), etc.)
- Common code patterns

Usage:
    python scripts/train_code_tokenizer.py
    python scripts/train_code_tokenizer.py --vocab-size 32000
"""

import argparse
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
import json


def create_code_tokenizer(vocab_size: int = 32000) -> Tokenizer:
    """Create a BPE tokenizer optimized for code."""

    # Use BPE model
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # Pre-tokenization: split on whitespace but preserve indentation
    # Use ByteLevel to handle all characters properly
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Decoder that properly handles ByteLevel encoding
    tokenizer.decoder = decoders.ByteLevel()

    # Post-processor for special tokens
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    return tokenizer


def get_training_files(data_dir: Path) -> list:
    """Get all training files."""
    files = []

    # Code corpus
    code_corpus = data_dir / "raw" / "code_corpus.txt"
    if code_corpus.exists():
        files.append(str(code_corpus))
        print(f"  Found: {code_corpus}")

    # Additional bash scripts
    bash_scripts = data_dir / "raw" / "additional_bash_scripts.txt"
    if bash_scripts.exists():
        files.append(str(bash_scripts))
        print(f"  Found: {bash_scripts}")

    # TinyStories for language capability
    tiny_stories = data_dir / "raw" / "TinyStoriesV2-GPT4-train.txt"
    if tiny_stories.exists():
        files.append(str(tiny_stories))
        print(f"  Found: {tiny_stories}")

    return files


def main():
    parser = argparse.ArgumentParser(description="Train code tokenizer")
    parser.add_argument("--vocab-size", type=int, default=32000,
                        help="Vocabulary size (default: 32000)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Data directory")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path (default: data/tokenizer/tokenizer.json)")
    parser.add_argument("--backup-old", action="store_true", default=True,
                        help="Backup old tokenizer before overwriting")
    args = parser.parse_args()

    print("=" * 60)
    print("Training Code Tokenizer")
    print("=" * 60)
    print(f"Vocabulary size: {args.vocab_size:,}")
    print()

    # Find training files
    print("Finding training files...")
    training_files = get_training_files(args.data_dir)

    if not training_files:
        print("\nError: No training files found!")
        print("Run these first:")
        print("  python scripts/download_code_corpus.py")
        return

    print(f"\nTraining on {len(training_files)} files")

    # Create tokenizer
    print("\nCreating tokenizer...")
    tokenizer = create_code_tokenizer(args.vocab_size)

    # Special tokens for code model
    special_tokens = [
        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>",
        "<sep>",
        "### Language:",
        "### Code:",
        "### End",
        "### Prompt:",
        "### Response:",
        "### Instruction:",
        "### Output:",
        "<tool_call>",
        "</tool_call>",
        "<function_results>",
        "</function_results>",
    ]

    # Create trainer
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
    )

    # Train
    print("\nTraining tokenizer...")
    tokenizer.train(training_files, trainer)

    # Set up output path
    output_path = args.output or args.data_dir / "tokenizer" / "tokenizer.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Backup old tokenizer if it exists
    if args.backup_old and output_path.exists():
        backup_path = output_path.with_suffix('.json.backup')
        print(f"\nBacking up old tokenizer to {backup_path}")
        import shutil
        shutil.copy(output_path, backup_path)

    # Save
    tokenizer.save(str(output_path))
    print(f"\n✓ Saved tokenizer to {output_path}")

    # Test the tokenizer
    print("\n" + "=" * 60)
    print("Testing Tokenizer")
    print("=" * 60)

    test_cases = [
        "#!/bin/bash",
        "def hello():",
        "for i in range(10):",
        'echo "Hello, World!"',
        "if [[ -f \"$file\" ]]; then",
        "import torch\nimport numpy as np",
        "SELECT * FROM users WHERE id = 1;",
        "docker run -d --name app -p 8080:80 myimage",
    ]

    # Reload tokenizer to test
    loaded = Tokenizer.from_file(str(output_path))

    print("\nTokenization tests:")
    for test in test_cases:
        encoding = loaded.encode(test)
        decoded = loaded.decode(encoding.ids)
        tokens = encoding.tokens[:8]  # Show first 8 tokens

        match = "✓" if decoded.strip() == test.strip() else "≈"
        print(f"\n  {match} '{test}'")
        print(f"    Tokens: {tokens}...")
        print(f"    Decoded: '{decoded}'")

    print("\n" + "=" * 60)
    print("Tokenizer Training Complete!")
    print("=" * 60)
    print(f"\nVocabulary size: {loaded.get_vocab_size():,}")
    print(f"Saved to: {output_path}")
    print("\n⚠️  IMPORTANT: You need to retrain the model from scratch!")
    print("The old model weights are incompatible with the new tokenizer.")
    print("\nNext steps:")
    print("  1. Retrain Stage 1: python scripts/train.py --stage language")
    print("  2. Retrain Stage 2: python scripts/train.py --stage code --checkpoint models/language_model_best.pth")


if __name__ == "__main__":
    main()
