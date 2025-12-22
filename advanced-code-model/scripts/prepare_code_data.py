#!/usr/bin/env python3
"""
Prepare Code Training Data

Tokenizes the downloaded code corpus and creates training/validation arrays.

Usage:
    python scripts/prepare_code_data.py
    python scripts/prepare_code_data.py --input data/raw/code_corpus.txt
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm
import random

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_code_corpus(corpus_file: Path) -> list:
    """Load code corpus and split into individual examples."""
    print(f"\nLoading corpus from {corpus_file}...")

    content = corpus_file.read_text(encoding="utf-8", errors="ignore")

    # Split by example markers
    examples = []
    current_example = []
    current_lang = "unknown"

    for line in content.split("\n"):
        if line.startswith("### Language:"):
            if current_example:
                examples.append({
                    "language": current_lang,
                    "code": "\n".join(current_example),
                })
            current_lang = line.replace("### Language:", "").strip()
            current_example = []
        elif line.startswith("### End"):
            if current_example:
                examples.append({
                    "language": current_lang,
                    "code": "\n".join(current_example),
                })
            current_example = []
        elif line.startswith("### Code:"):
            continue  # Skip this marker
        elif line.startswith("### Description:"):
            continue  # Skip description
        else:
            current_example.append(line)

    # Don't forget the last example
    if current_example:
        examples.append({
            "language": current_lang,
            "code": "\n".join(current_example),
        })

    print(f"  Loaded {len(examples):,} examples")
    return examples


def tokenize_examples(
    examples: list,
    tokenizer: Tokenizer,
    max_length: int = 1024,
    min_tokens: int = 50,
) -> np.ndarray:
    """Tokenize examples into fixed-length sequences."""
    print(f"\nTokenizing {len(examples):,} examples...")

    all_sequences = []
    skipped = 0

    for example in tqdm(examples, desc="Tokenizing"):
        code = example.get("code", "")
        lang = example.get("language", "unknown")

        # Skip very short examples
        if len(code.strip()) < 50:
            skipped += 1
            continue

        # Format with language header
        text = f"### Language: {lang}\n### Code:\n{code}\n### End"

        # Tokenize
        encoding = tokenizer.encode(text)
        tokens = encoding.ids

        # Skip if too short after tokenization
        if len(tokens) < min_tokens:
            skipped += 1
            continue

        # Create fixed-length sequences (chunking long examples)
        for i in range(0, len(tokens), max_length):
            chunk = tokens[i:i + max_length]

            # Pad if needed
            if len(chunk) < max_length:
                chunk = chunk + [0] * (max_length - len(chunk))

            all_sequences.append(chunk)

    print(f"  Created {len(all_sequences):,} sequences")
    print(f"  Skipped {skipped:,} examples (too short)")

    return np.array(all_sequences, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(description="Prepare code training data")

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/code_corpus.txt"),
        help="Input corpus file"
    )

    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("data/tokenizer/tokenizer.json"),
        help="Tokenizer path"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory"
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum sequence length (default: 1024)"
    )

    parser.add_argument(
        "--val-split",
        type=float,
        default=0.05,
        help="Validation split ratio (default: 0.05)"
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="code_train_large",
        help="Output file base name (default: code_train_large)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Preparing Code Training Data")
    print("=" * 60)

    # Check inputs
    if not args.input.exists():
        print(f"Error: Corpus file not found: {args.input}")
        print("Run: python scripts/download_code_corpus.py first")
        sys.exit(1)

    if not args.tokenizer.exists():
        print(f"Error: Tokenizer not found: {args.tokenizer}")
        print("Run: python scripts/train_tokenizer.py first")
        sys.exit(1)

    # Load tokenizer
    print(f"\nLoading tokenizer from {args.tokenizer}...")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    print(f"  Vocabulary size: {tokenizer.get_vocab_size():,}")

    # Load corpus
    examples = load_code_corpus(args.input)

    # Shuffle
    random.shuffle(examples)

    # Tokenize
    sequences = tokenize_examples(examples, tokenizer, args.max_length)

    # Shuffle again
    np.random.shuffle(sequences)

    # Split into train/val
    split_idx = int(len(sequences) * (1 - args.val_split))
    train_data = sequences[:split_idx]
    val_data = sequences[split_idx:]

    print(f"\nDataset split:")
    print(f"  Training: {len(train_data):,} sequences")
    print(f"  Validation: {len(val_data):,} sequences")

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_file = args.output_dir / f"{args.output_name}.npy"
    val_file = args.output_dir / f"{args.output_name.replace('train', 'val')}.npy"

    np.save(train_file, train_data)
    np.save(val_file, val_data)

    print(f"\n✓ Saved training data: {train_file}")
    print(f"✓ Saved validation data: {val_file}")

    # Statistics
    train_tokens = train_data.shape[0] * train_data.shape[1]
    val_tokens = val_data.shape[0] * val_data.shape[1]
    train_mb = train_data.nbytes / (1024 * 1024)
    val_mb = val_data.nbytes / (1024 * 1024)

    print(f"\nDataset Statistics:")
    print(f"  Training: {train_data.shape[0]:,} sequences, {train_tokens:,} tokens ({train_mb:.1f} MB)")
    print(f"  Validation: {val_data.shape[0]:,} sequences, {val_tokens:,} tokens ({val_mb:.1f} MB)")

    # Print training command
    print("\n" + "=" * 60)
    print("Training Command")
    print("=" * 60)
    print(f"""
python scripts/train.py \\
    --stage code \\
    --checkpoint models/language_model_best.pth \\
    --data-file {args.output_name}.npy \\
    --model-size large \\
    --batch-size 2 \\
    --num-epochs 5 \\
    --learning-rate 5e-6 \\
    --warmup-steps 500 \\
    --use-rmsnorm --use-rope --use-compile --use-amp

Or to continue from current checkpoint:

python scripts/train.py \\
    --stage code \\
    --checkpoint models/code_model_best.pth \\
    --data-file {args.output_name}.npy \\
    --model-size large \\
    --batch-size 2 \\
    --num-epochs 3 \\
    --learning-rate 3e-6 \\
    --warmup-steps 200 \\
    --use-rmsnorm --use-rope --use-compile --use-amp
""")


if __name__ == "__main__":
    main()
