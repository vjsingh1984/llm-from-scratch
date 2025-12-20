"""
Download BookCorpus dataset for language pretraining.

BookCorpus: 11,000 free books, ~5GB of high-quality English text.
Perfect for pretraining language models before fine-tuning on code.
"""

import sys
from pathlib import Path
import json
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_bookcorpus(output_dir: Path, max_samples: int = None):
    """
    Download BookCorpus dataset using HuggingFace datasets.

    Args:
        output_dir: Where to save the data
        max_samples: Maximum number of samples (None = all)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library not installed")
        print("Install with: pip install datasets")
        return None

    print("="*60)
    print("Downloading BookCorpus Dataset")
    print("="*60)
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Download dataset
    print("Loading BookCorpus from HuggingFace...")
    print("Note: First download may take 10-20 minutes (5GB)")
    print()

    try:
        # Load dataset - this will cache locally
        dataset = load_dataset("bookcorpus", split="train")

        print(f"Dataset loaded: {len(dataset)} samples")

        # Limit samples if specified
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            print(f"Using {len(dataset)} samples (limited)")

        # Extract texts
        print("\nExtracting texts...")
        texts = []

        for item in tqdm(dataset):
            text = item['text'].strip()
            if len(text) > 50:  # Filter very short texts
                texts.append(text)

        print(f"\nExtracted {len(texts)} text samples")

        # Save as JSON
        json_file = output_dir / "bookcorpus.json"
        print(f"\nSaving to {json_file}...")

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'texts': texts,
                'count': len(texts),
                'source': 'bookcorpus'
            }, f, indent=2)

        # Save as plain text (for inspection)
        txt_file = output_dir / "bookcorpus.txt"
        print(f"Saving to {txt_file}...")

        with open(txt_file, 'w', encoding='utf-8') as f:
            for text in texts[:1000]:  # First 1000 for inspection
                f.write(text)
                f.write("\n\n" + "="*60 + "\n\n")

        # Statistics
        total_chars = sum(len(t) for t in texts)
        total_words = sum(len(t.split()) for t in texts)

        stats = {
            'num_texts': len(texts),
            'total_chars': total_chars,
            'total_words': total_words,
            'avg_chars_per_text': total_chars / len(texts),
            'avg_words_per_text': total_words / len(texts),
        }

        stats_file = output_dir / "bookcorpus_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print("\n" + "="*60)
        print("Download Complete!")
        print("="*60)
        print(f"\nStatistics:")
        print(f"  Texts: {stats['num_texts']:,}")
        print(f"  Total characters: {stats['total_chars']:,}")
        print(f"  Total words: {stats['total_words']:,}")
        print(f"  Avg length: {stats['avg_chars_per_text']:.0f} chars, {stats['avg_words_per_text']:.0f} words")

        print(f"\nFiles saved:")
        print(f"  JSON: {json_file}")
        print(f"  Text (sample): {txt_file}")
        print(f"  Stats: {stats_file}")

        return texts

    except Exception as e:
        print(f"\nError downloading BookCorpus: {e}")
        print("\nAlternative: Use TinyStories (already downloaded)")
        print("  Location: ../llm-from-scratch/data/tiny_stories.txt")
        return None


def check_existing_data(data_dir: Path):
    """Check what language data we already have."""
    print("="*60)
    print("Checking Existing Language Data")
    print("="*60)
    print()

    sources = []

    # Check for TinyStories
    tinystories = data_dir.parent.parent / 'llm-from-scratch' / 'data' / 'tiny_stories.txt'
    if tinystories.exists():
        size = tinystories.stat().st_size / (1024 * 1024)  # MB
        sources.append({
            'name': 'TinyStories',
            'path': tinystories,
            'size_mb': size,
            'quality': 'High (synthetic, clean)'
        })
        print(f"✓ TinyStories found: {size:.1f}MB")

    # Check for BookCorpus
    bookcorpus = data_dir / 'bookcorpus.json'
    if bookcorpus.exists():
        with open(bookcorpus) as f:
            data = json.load(f)
            size = bookcorpus.stat().st_size / (1024 * 1024)
            sources.append({
                'name': 'BookCorpus',
                'path': bookcorpus,
                'size_mb': size,
                'count': data['count'],
                'quality': 'High (books)'
            })
            print(f"✓ BookCorpus found: {size:.1f}MB, {data['count']:,} samples")

    if not sources:
        print("✗ No language data found")
        print("\nRecommendation: Download BookCorpus or use TinyStories")

    print()
    return sources


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Download BookCorpus')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent.parent / 'data_language',
                        help='Output directory')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to download (default: all)')
    parser.add_argument('--check-only', action='store_true',
                        help='Only check existing data, do not download')

    args = parser.parse_args()

    # Check existing data
    existing = check_existing_data(args.output_dir)

    if args.check_only:
        return

    # Ask user if they want to download
    if existing:
        print("Found existing data. Download BookCorpus anyway?")
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Using existing data.")
            return

    # Download
    download_bookcorpus(args.output_dir, args.max_samples)


if __name__ == '__main__':
    main()
