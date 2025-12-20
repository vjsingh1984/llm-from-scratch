"""
Download and prepare BookCorpus dataset for language pretraining.

BookCorpus is a large corpus of 11,000+ books used for pretraining
language models. This script downloads the dataset from Hugging Face
and prepares it for training.

Usage:
    python scripts/download_bookcorpus.py --output data/bookcorpus

Expected size: ~5GB
Expected time: 30-60 minutes (depends on internet speed)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
from tqdm import tqdm


def download_bookcorpus(output_dir: Path, num_proc: int = 4) -> Dict[str, int]:
    """
    Download BookCorpus dataset from Hugging Face.

    Args:
        output_dir: Directory to save the dataset
        num_proc: Number of parallel processes for downloading

    Returns:
        Statistics dictionary
    """
    print("=" * 60)
    print("Downloading BookCorpus Dataset")
    print("=" * 60)
    print()

    # Create output directory
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Processes: {num_proc}")
    print()

    # Download dataset
    print("Downloading from Hugging Face...")
    print("This may take 30-60 minutes depending on your connection.")
    print()

    try:
        dataset = load_dataset(
            "bookcorpus",
            split="train",
            cache_dir=str(raw_dir),
            num_proc=num_proc
        )
    except Exception as e:
        print(f"Error downloading from Hugging Face: {e}")
        print()
        print("Trying alternative source...")

        # Alternative: Use books3 subset from ThePile
        dataset = load_dataset(
            "EleutherAI/pile-uncopyrighted",
            split="train",
            cache_dir=str(raw_dir),
            streaming=False
        )

        # Filter for books
        print("Filtering for book content...")
        dataset = dataset.filter(
            lambda x: x.get("meta", {}).get("pile_set_name") == "Books3",
            num_proc=num_proc
        )

    print(f"✓ Downloaded {len(dataset):,} documents")
    print()

    # Save as text files
    print("Saving as text files...")
    texts = []
    total_chars = 0
    total_words = 0

    for idx, example in enumerate(tqdm(dataset, desc="Processing")):
        text = example["text"]
        texts.append(text)

        total_chars += len(text)
        total_words += len(text.split())

        # Save in batches to manage memory
        if (idx + 1) % 1000 == 0:
            batch_file = raw_dir / f"batch_{(idx + 1) // 1000:04d}.txt"
            with open(batch_file, "w", encoding="utf-8") as f:
                f.write("\n\n".join(texts))
            texts = []

    # Save remaining texts
    if texts:
        batch_file = raw_dir / f"batch_final.txt"
        with open(batch_file, "w", encoding="utf-8") as f:
            f.write("\n\n".join(texts))

    print()
    print("✓ Saved all text files")
    print()

    # Calculate statistics
    stats = {
        "num_documents": len(dataset),
        "total_characters": total_chars,
        "total_words": total_words,
        "avg_chars_per_doc": total_chars / len(dataset),
        "avg_words_per_doc": total_words / len(dataset),
        "estimated_tokens": total_words * 1.3,  # Approximate BPE tokens
        "size_gb": total_chars / (1024 ** 3),
    }

    # Save statistics
    stats_file = output_dir / "stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print("Dataset Statistics:")
    print(f"  Documents: {stats['num_documents']:,}")
    print(f"  Total characters: {stats['total_characters']:,}")
    print(f"  Total words: {stats['total_words']:,}")
    print(f"  Estimated tokens: {stats['estimated_tokens']:,.0f}")
    print(f"  Size: {stats['size_gb']:.2f} GB")
    print(f"  Avg words/doc: {stats['avg_words_per_doc']:.0f}")
    print()
    print(f"✓ Statistics saved to {stats_file}")

    return stats


def verify_download(output_dir: Path) -> bool:
    """
    Verify that the download was successful.

    Args:
        output_dir: Directory containing the dataset

    Returns:
        True if verification passes
    """
    print()
    print("Verifying download...")

    raw_dir = output_dir / "raw"

    if not raw_dir.exists():
        print("✗ Raw directory not found")
        return False

    text_files = list(raw_dir.glob("batch_*.txt"))
    if not text_files:
        print("✗ No text files found")
        return False

    stats_file = output_dir / "stats.json"
    if not stats_file.exists():
        print("✗ Statistics file not found")
        return False

    # Check file sizes
    total_size = sum(f.stat().st_size for f in text_files)
    size_gb = total_size / (1024 ** 3)

    if size_gb < 3.0:
        print(f"⚠ Warning: Dataset size ({size_gb:.2f} GB) is smaller than expected (5GB)")
        print("  Dataset may be incomplete")
        return False

    print(f"✓ Found {len(text_files)} text files")
    print(f"✓ Total size: {size_gb:.2f} GB")
    print(f"✓ Statistics file present")
    print()
    print("✓ Verification passed!")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download BookCorpus dataset for language pretraining"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/bookcorpus"),
        help="Output directory (default: data/bookcorpus)"
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=4,
        help="Number of parallel processes (default: 4)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing download"
    )

    args = parser.parse_args()

    if args.verify_only:
        success = verify_download(args.output)
        exit(0 if success else 1)

    # Download dataset
    stats = download_bookcorpus(args.output, args.num_proc)

    # Verify
    success = verify_download(args.output)

    if success:
        print()
        print("=" * 60)
        print("BookCorpus download complete!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  1. Tokenize: python scripts/prepare_bookcorpus.py")
        print("  2. Train: python scripts/train_pretrain.py")
        exit(0)
    else:
        print()
        print("✗ Verification failed. Please try downloading again.")
        exit(1)


if __name__ == "__main__":
    main()
