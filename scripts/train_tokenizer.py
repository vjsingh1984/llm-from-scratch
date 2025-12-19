"""
Train the BPE tokenizer on TinyStories dataset.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer import BPETokenizer


def load_text_data(file_path: Path) -> list[str]:
    """
    Load text data from file.

    Args:
        file_path: Path to text file

    Returns:
        List of text segments (split by <|endoftext|>)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split on end-of-text token
    segments = text.split('<|endoftext|>')

    # Filter out empty segments
    segments = [s.strip() for s in segments if s.strip()]

    return segments


def train_tokenizer(
    data_file: Path,
    output_dir: Path,
    vocab_size: int = 32000,
    max_samples: int = None
):
    """
    Train BPE tokenizer on data.

    Args:
        data_file: Path to training data file
        output_dir: Directory to save trained tokenizer
        vocab_size: Target vocabulary size
        max_samples: Maximum number of samples to use (None = use all)
    """
    print(f"Loading data from {data_file}...")
    texts = load_text_data(data_file)

    if max_samples:
        texts = texts[:max_samples]

    print(f"Loaded {len(texts)} text segments")
    print(f"Total characters: {sum(len(t) for t in texts):,}")

    # Create tokenizer
    print(f"\nInitializing tokenizer with vocab size {vocab_size}...")
    tokenizer = BPETokenizer(vocab_size=vocab_size)

    # Train
    print("\nTraining tokenizer...")
    tokenizer.train(texts, verbose=True)

    # Save
    output_dir = Path(output_dir)
    print(f"\nSaving tokenizer to {output_dir}...")
    tokenizer.save(output_dir)

    # Test tokenizer
    print("\n" + "="*60)
    print("Testing tokenizer")
    print("="*60)

    test_texts = [
        "Once upon a time, there was a little girl named Lily.",
        "The quick brown fox jumps over the lazy dog.",
        "In a galaxy far, far away...",
        "Hello, world! How are you today?",
    ]

    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        print(f"\nOriginal: {text}")
        print(f"Encoded:  {encoded[:20]}{'...' if len(encoded) > 20 else ''}")
        print(f"Num tokens: {len(encoded)}")
        print(f"Decoded:  {decoded}")
        print(f"Match: {'✓' if text == decoded else '✗ MISMATCH'}")

    print("\n" + "="*60)
    print("Tokenizer training complete!")
    print("="*60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train BPE tokenizer')
    parser.add_argument(
        '--data-file',
        type=str,
        default='../data/tinystories_10000.txt',
        help='Path to training data file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../tokenizer_model',
        help='Directory to save trained tokenizer'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=8000,
        help='Target vocabulary size (default: 8000 for quick training)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to use (default: all)'
    )

    args = parser.parse_args()

    # Get absolute paths
    script_dir = Path(__file__).parent
    data_file = (script_dir / args.data_file).resolve()
    output_dir = (script_dir / args.output_dir).resolve()

    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        print("Please run download_data.py first!")
        sys.exit(1)

    train_tokenizer(
        data_file=data_file,
        output_dir=output_dir,
        vocab_size=args.vocab_size,
        max_samples=args.max_samples
    )
