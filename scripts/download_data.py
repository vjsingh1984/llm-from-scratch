"""
Download and prepare the TinyStories dataset for training.

TinyStories is a dataset of simple stories written in clear language,
perfect for training small language models.
"""

import sys
from pathlib import Path

# Add parent directory to path to import tokenizer
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from datasets import load_dataset
except ImportError:
    print("Please install datasets: pip install datasets")
    sys.exit(1)


def download_tinystories(output_dir: Path, num_samples: int = 10000):
    """
    Download TinyStories dataset and save to text file.

    Args:
        output_dir: Directory to save data
        num_samples: Number of samples to download (default: 10000 for quick start)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading TinyStories dataset ({num_samples} samples)...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    # Take subset
    dataset = dataset.select(range(min(num_samples, len(dataset))))

    print(f"Processing {len(dataset)} stories...")

    # Combine stories with EOS token
    stories = []
    for item in dataset:
        stories.append(item['text'].strip())

    # Join with end-of-text token
    combined_text = '<|endoftext|>'.join(stories)

    # Save to file
    output_file = output_dir / f'tinystories_{num_samples}.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(combined_text)

    print(f"\nData saved to: {output_file}")
    print(f"Total characters: {len(combined_text):,}")
    print(f"Total stories: {len(stories):,}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

    # Also save validation set
    print("\nDownloading validation set...")
    val_dataset = load_dataset("roneneldan/TinyStories", split="validation")
    val_dataset = val_dataset.select(range(min(1000, len(val_dataset))))

    val_stories = [item['text'].strip() for item in val_dataset]
    val_text = '<|endoftext|>'.join(val_stories)

    val_file = output_dir / 'tinystories_validation.txt'
    with open(val_file, 'w', encoding='utf-8') as f:
        f.write(val_text)

    print(f"Validation data saved to: {val_file}")
    print(f"Validation stories: {len(val_stories):,}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Download TinyStories dataset')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../data',
        help='Output directory for data files'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10000,
        help='Number of samples to download (default: 10000)'
    )

    args = parser.parse_args()

    # Get absolute path relative to script location
    script_dir = Path(__file__).parent
    output_dir = (script_dir / args.output_dir).resolve()

    download_tinystories(output_dir, args.num_samples)
    print("\nDone! You can now train the tokenizer.")
