"""
Train BPE tokenizer on combined language + code corpus.

This script trains a Byte-Pair Encoding (BPE) tokenizer on both:
1. Language data (TinyStories - 2.1M documents)
2. Code data (Bash scripts - 1,767 scripts)

Target vocabulary: 32,000 tokens
Special tokens: <PAD>, <UNK>, <BOS>, <EOS>

Usage:
    python scripts/train_tokenizer.py
"""

import json
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm


def iter_language_data(data_dir: Path) -> Iterator[str]:
    """
    Iterate over language corpus (TinyStories).

    Args:
        data_dir: Path to language data directory

    Yields:
        Text content from each batch file
    """
    raw_dir = data_dir / "raw"

    batch_files = sorted(raw_dir.glob("batch_*.txt"))

    print(f"Found {len(batch_files)} language batch files")

    for batch_file in tqdm(batch_files, desc="Loading language data"):
        try:
            with open(batch_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Split into individual stories
            stories = content.strip().split("\n\n")

            for story in stories:
                if story.strip():
                    yield story.strip()

        except Exception as e:
            print(f"Error reading {batch_file}: {e}")
            continue


def iter_code_data(data_dir: Path) -> Iterator[str]:
    """
    Iterate over code corpus (Bash scripts).

    Args:
        data_dir: Path to bash scripts directory

    Yields:
        Script content from each .sh file
    """
    scripts_dir = data_dir / "raw" / "scripts"

    script_files = sorted(scripts_dir.glob("*.sh"))

    print(f"Found {len(script_files)} bash scripts")

    for script_file in tqdm(script_files, desc="Loading code data"):
        try:
            with open(script_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            if content.strip():
                yield content.strip()

        except Exception as e:
            print(f"Error reading {script_file}: {e}")
            continue


def train_tokenizer(
    language_dir: Path,
    code_dir: Path,
    output_dir: Path,
    vocab_size: int = 32000
) -> Tokenizer:
    """
    Train BPE tokenizer on combined corpus.

    Args:
        language_dir: Path to language data
        code_dir: Path to code data
        output_dir: Output directory for tokenizer
        vocab_size: Target vocabulary size

    Returns:
        Trained tokenizer
    """
    print("=" * 60)
    print("Training BPE Tokenizer")
    print("=" * 60)
    print()
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Language data: {language_dir}")
    print(f"Code data: {code_dir}")
    print()

    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = Whitespace()

    # Configure trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
        show_progress=True,
    )

    # Collect training data
    print("Collecting training data...")
    print()

    training_data = []

    # Add language data
    for text in iter_language_data(language_dir):
        training_data.append(text)

    language_count = len(training_data)
    print(f"✓ Loaded {language_count:,} language examples")
    print()

    # Add code data
    code_start = len(training_data)
    for text in iter_code_data(code_dir):
        training_data.append(text)

    code_count = len(training_data) - code_start
    print(f"✓ Loaded {code_count:,} code examples")
    print()

    total_count = len(training_data)
    print(f"Total training examples: {total_count:,}")
    print(f"  Language: {language_count:,} ({100*language_count/total_count:.1f}%)")
    print(f"  Code: {code_count:,} ({100*code_count/total_count:.1f}%)")
    print()

    # Train tokenizer
    print("Training tokenizer...")
    print("This may take 10-20 minutes for large corpora.")
    print()

    tokenizer.train_from_iterator(training_data, trainer=trainer)

    print()
    print("✓ Tokenizer training complete!")
    print()

    # Save tokenizer
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_file = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_file))

    print(f"✓ Saved tokenizer to {tokenizer_file}")

    # Test tokenizer
    print()
    print("Testing tokenizer...")
    print()

    test_cases = [
        "Hello, world! This is a test.",
        '#!/bin/bash\necho "Starting deployment..."',
        "for i in range(10):",
    ]

    for test_text in test_cases:
        encoded = tokenizer.encode(test_text)
        print(f"Input:  {test_text}")
        print(f"Tokens: {encoded.tokens}")
        print(f"IDs:    {encoded.ids}")
        print()

    # Save metadata
    vocab = tokenizer.get_vocab()

    metadata = {
        "vocab_size": len(vocab),
        "training_examples": total_count,
        "language_examples": language_count,
        "code_examples": code_count,
        "special_tokens": ["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
    }

    metadata_file = output_dir / "tokenizer_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved metadata to {metadata_file}")
    print()

    # Print statistics
    print("Tokenizer Statistics:")
    print(f"  Vocabulary size: {len(vocab):,}")
    print(f"  Training examples: {total_count:,}")
    print(f"  Special tokens: {metadata['special_tokens']}")
    print()

    return tokenizer


def main():
    """Main entry point."""
    # Paths
    base_dir = Path(__file__).parent.parent
    language_dir = base_dir / "data" / "language"
    code_dir = base_dir / "data" / "bash"
    output_dir = base_dir / "data" / "tokenizer"

    # Train tokenizer
    tokenizer = train_tokenizer(
        language_dir=language_dir,
        code_dir=code_dir,
        output_dir=output_dir,
        vocab_size=32000
    )

    print("=" * 60)
    print("Tokenizer training complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Prepare datasets: python scripts/prepare_datasets.py")
    print("  2. Train model: python scripts/train.py")


if __name__ == "__main__":
    main()
