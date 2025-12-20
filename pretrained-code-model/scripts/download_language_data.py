"""
Download language data for pretraining.

Options:
1. BookCorpus (11K books, ~5GB) - Best quality
2. TinyStories (5K stories, ~430MB) - Fast, already downloaded
3. Wikipedia (70GB) - Very large, high quality
"""

import sys
from pathlib import Path
import json
from tqdm import tqdm
import argparse

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_bookcorpus(output_dir: Path, max_samples: int = None):
    """Download BookCorpus dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library not installed")
        print("Install with: pip install datasets")
        return False

    print("="*60)
    print("Downloading BookCorpus")
    print("="*60)
    print()
    print("Size: ~5GB")
    print("Content: 11,000 free books")
    print("Quality: High (natural book text)")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("Loading from HuggingFace (may take 10-20 min first time)...")
        dataset = load_dataset("bookcorpus", split="train")

        print(f"Dataset loaded: {len(dataset):,} samples")

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            print(f"Limited to: {len(dataset):,} samples")

        # Extract texts
        print("\nProcessing texts...")
        texts = []

        for item in tqdm(dataset, desc="Extracting"):
            text = item['text'].strip()
            if len(text) > 50:  # Filter very short
                texts.append(text)

        # Save
        print(f"\nSaving {len(texts):,} texts...")

        # JSON format
        json_file = output_dir / "language_data.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'texts': texts,
                'count': len(texts),
                'source': 'bookcorpus'
            }, f, indent=2)

        # Statistics
        total_chars = sum(len(t) for t in texts)
        total_words = sum(len(t.split()) for t in texts)

        stats = {
            'source': 'bookcorpus',
            'num_texts': len(texts),
            'total_chars': total_chars,
            'total_words': total_words,
            'avg_chars': total_chars / len(texts) if texts else 0,
            'avg_words': total_words / len(texts) if texts else 0,
        }

        stats_file = output_dir / "stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print("\n" + "="*60)
        print("BookCorpus Downloaded!")
        print("="*60)
        print(f"  Texts: {stats['num_texts']:,}")
        print(f"  Characters: {stats['total_chars']:,}")
        print(f"  Words: {stats['total_words']:,}")
        print(f"  Avg length: {stats['avg_words']:.0f} words")
        print(f"\nSaved to: {json_file}")

        return True

    except Exception as e:
        print(f"\nError downloading BookCorpus: {e}")
        return False


def use_tinystories(output_dir: Path):
    """Use TinyStories from LLM project."""
    tinystories_file = Path(__file__).parent.parent.parent / 'data' / 'tinystories_5000.txt'

    if not tinystories_file.exists():
        print(f"Error: TinyStories not found at {tinystories_file}")
        print("Please run the LLM project first to download it.")
        return False

    print("="*60)
    print("Using TinyStories")
    print("="*60)
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Read TinyStories
    print(f"Reading from {tinystories_file}...")
    with open(tinystories_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split into stories (separated by double newlines)
    texts = [s.strip() for s in content.split('\n\n') if s.strip() and len(s) > 50]

    print(f"Found {len(texts):,} stories")

    # Save to our format
    json_file = output_dir / "language_data.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'texts': texts,
            'count': len(texts),
            'source': 'tinystories'
        }, f, indent=2)

    # Stats
    total_chars = sum(len(t) for t in texts)
    total_words = sum(len(t.split()) for t in texts)

    stats = {
        'source': 'tinystories',
        'num_texts': len(texts),
        'total_chars': total_chars,
        'total_words': total_words,
        'avg_chars': total_chars / len(texts),
        'avg_words': total_words / len(texts),
    }

    stats_file = output_dir / "stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "="*60)
    print("TinyStories Prepared!")
    print("="*60)
    print(f"  Stories: {stats['num_texts']:,}")
    print(f"  Characters: {stats['total_chars']:,}")
    print(f"  Words: {stats['total_words']:,}")
    print(f"  Avg length: {stats['avg_words']:.0f} words")
    print(f"\nSaved to: {json_file}")

    return True


def check_existing(data_dir: Path):
    """Check if data already exists."""
    json_file = data_dir / "language_data.json"
    stats_file = data_dir / "stats.json"

    if json_file.exists() and stats_file.exists():
        with open(stats_file) as f:
            stats = json.load(f)

        print("="*60)
        print("Existing Language Data Found!")
        print("="*60)
        print(f"  Source: {stats['source']}")
        print(f"  Texts: {stats['num_texts']:,}")
        print(f"  Words: {stats['total_words']:,}")
        print(f"  Location: {json_file}")
        print()

        return True

    return False


def main():
    parser = argparse.ArgumentParser(description='Download language pretraining data')

    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent.parent / 'data_language',
                        help='Output directory')
    parser.add_argument('--source', type=str,
                        choices=['bookcorpus', 'tinystories', 'auto'],
                        default='auto',
                        help='Data source (auto = check existing, then TinyStories)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit samples (for testing)')
    parser.add_argument('--force', action='store_true',
                        help='Force download even if data exists')

    args = parser.parse_args()

    print("\n")
    print("="*60)
    print("Language Data Download")
    print("="*60)
    print()

    # Check existing
    if not args.force and check_existing(args.output_dir):
        response = input("Data exists. Download anyway? [y/N]: ")
        if response.lower() != 'y':
            print("Using existing data.")
            return

    # Download based on source
    success = False

    if args.source == 'bookcorpus':
        success = download_bookcorpus(args.output_dir, args.max_samples)

    elif args.source == 'tinystories':
        success = use_tinystories(args.output_dir)

    else:  # auto
        # Try TinyStories first (faster, already have it)
        print("Auto mode: Trying TinyStories first (faster)...")
        print()
        success = use_tinystories(args.output_dir)

        if not success:
            print("\nTinyStories not available. Try BookCorpus? (5GB download)")
            response = input("Download BookCorpus? [y/N]: ")
            if response.lower() == 'y':
                success = download_bookcorpus(args.output_dir, args.max_samples)

    if success:
        print("\n" + "="*60)
        print("Ready for Stage 1: Language Pretraining")
        print("="*60)
        print("\nNext step:")
        print("  python scripts/train_language.py")
    else:
        print("\nFailed to download language data.")
        print("\nOptions:")
        print("  1. Use TinyStories: --source tinystories")
        print("  2. Use BookCorpus: --source bookcorpus")


if __name__ == '__main__':
    main()
