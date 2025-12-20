"""
Download sample data for quick testing and demonstration.

This creates a small-scale version of the full pipeline:
- Sample text data (~100MB instead of 5GB)
- Sample bash scripts (~50 instead of 10K+)

Usage:
    python scripts/download_sample_data.py --output data/sample
"""

import argparse
import json
from pathlib import Path
from typing import Dict

from datasets import load_dataset
from tqdm import tqdm


def download_sample_text(output_dir: Path, num_examples: int = 1000) -> Dict:
    """Download sample text data for language pretraining."""
    print("=" * 60)
    print("Downloading Sample Text Data")
    print("=" * 60)
    print()

    raw_dir = output_dir / "text" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {num_examples} text samples...")
    print("(Full dataset would be 11K books, ~5GB)")
    print()

    # Download TinyStories as a substitute (smaller, faster)
    dataset = load_dataset(
        "roneneldan/TinyStories",
        split=f"train[:{num_examples}]",
        cache_dir=str(raw_dir)
    )

    print(f"✓ Downloaded {len(dataset):,} examples")
    print()

    # Save as text
    print("Saving text files...")
    texts = []
    total_chars = 0
    total_words = 0

    for idx, example in enumerate(tqdm(dataset, desc="Processing")):
        text = example["text"]
        texts.append(text)
        total_chars += len(text)
        total_words += len(text.split())

    # Save all texts
    text_file = raw_dir / "sample_text.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(texts))

    stats = {
        "num_documents": len(texts),
        "total_characters": total_chars,
        "total_words": total_words,
        "estimated_tokens": int(total_words * 1.3),
        "size_mb": total_chars / (1024 ** 2),
    }

    # Save stats
    stats_file = output_dir / "text" / "stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print()
    print("Sample Data Statistics:")
    print(f"  Documents: {stats['num_documents']:,}")
    print(f"  Words: {stats['total_words']:,}")
    print(f"  Estimated tokens: {stats['estimated_tokens']:,}")
    print(f"  Size: {stats['size_mb']:.2f} MB")
    print()
    print(f"✓ Saved to {text_file}")
    print(f"✓ Stats saved to {stats_file}")

    return stats


def create_sample_bash_scripts(output_dir: Path, num_scripts: int = 50) -> Dict:
    """Create sample bash scripts."""
    print()
    print("=" * 60)
    print("Creating Sample Bash Scripts")
    print("=" * 60)
    print()

    scripts_dir = output_dir / "bash" / "raw" / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating {num_scripts} bash script samples...")
    print("(Full corpus would be 10K+ scripts from GitHub)")
    print()

    # Generate diverse bash scripts
    script_templates = [
        # System administration
        '''#!/bin/bash
# System backup script

set -euo pipefail

BACKUP_DIR="/backup"
SOURCE_DIR="/data"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Starting backup at $TIMESTAMP"
tar -czf "$BACKUP_DIR/backup_$TIMESTAMP.tar.gz" "$SOURCE_DIR"
echo "Backup complete"
''',
        # DevOps
        '''#!/bin/bash
# Docker cleanup script

echo "Cleaning up Docker resources..."

# Remove stopped containers
docker container prune -f

# Remove unused images
docker image prune -a -f

# Remove unused volumes
docker volume prune -f

echo "Cleanup complete"
''',
        # Monitoring
        '''#!/bin/bash
# System monitoring

while true; do
    clear
    echo "=== System Monitor ==="
    echo
    echo "CPU Usage:"
    top -l 1 | grep "CPU usage"
    echo
    echo "Memory:"
    top -l 1 | grep "PhysMem"
    echo
    echo "Disk:"
    df -h /
    sleep 5
done
''',
        # Git automation
        '''#!/bin/bash
# Git auto-commit

if [[ -z $(git status -s) ]]; then
    echo "No changes to commit"
    exit 0
fi

git add .
git commit -m "Auto-commit: $(date)"
git push origin main

echo "Changes committed and pushed"
''',
        # Log processing
        '''#!/bin/bash
# Process application logs

LOG_FILE="/var/log/app.log"
ERRORS=$(grep "ERROR" "$LOG_FILE" | wc -l)
WARNINGS=$(grep "WARN" "$LOG_FILE" | wc -l)

echo "Log Summary:"
echo "  Errors: $ERRORS"
echo "  Warnings: $WARNINGS"

if [[ $ERRORS -gt 0 ]]; then
    echo "Recent errors:"
    grep "ERROR" "$LOG_FILE" | tail -5
fi
''',
    ]

    saved_scripts = []
    total_size = 0
    total_lines = 0

    for i in range(num_scripts):
        # Cycle through templates with variations
        template = script_templates[i % len(script_templates)]

        # Add some variation
        script = template

        script_file = scripts_dir / f"script_{i+1:03d}.sh"
        with open(script_file, "w") as f:
            f.write(script)

        saved_scripts.append({
            "id": i + 1,
            "file": script_file.name,
            "size": len(script),
            "lines": script.count("\n") + 1,
        })

        total_size += len(script)
        total_lines += script.count("\n") + 1

    stats = {
        "num_scripts": len(saved_scripts),
        "total_characters": total_size,
        "total_lines": total_lines,
        "size_kb": total_size / 1024,
        "avg_lines_per_script": total_lines / len(saved_scripts),
    }

    # Save metadata
    metadata_file = output_dir / "bash" / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump({"stats": stats, "scripts": saved_scripts}, f, indent=2)

    stats_file = output_dir / "bash" / "stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print("Sample Scripts Statistics:")
    print(f"  Scripts: {stats['num_scripts']:,}")
    print(f"  Total lines: {stats['total_lines']:,}")
    print(f"  Size: {stats['size_kb']:.2f} KB")
    print(f"  Avg lines/script: {stats['avg_lines_per_script']:.0f}")
    print()
    print(f"✓ Saved to {scripts_dir}")
    print(f"✓ Metadata saved to {metadata_file}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download sample data for testing"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sample"),
        help="Output directory (default: data/sample)"
    )
    parser.add_argument(
        "--num-texts",
        type=int,
        default=1000,
        help="Number of text samples (default: 1000)"
    )
    parser.add_argument(
        "--num-scripts",
        type=int,
        default=50,
        help="Number of bash scripts (default: 50)"
    )

    args = parser.parse_args()

    # Download sample text
    text_stats = download_sample_text(args.output, args.num_texts)

    # Create sample bash scripts
    bash_stats = create_sample_bash_scripts(args.output, args.num_scripts)

    print()
    print("=" * 60)
    print("Sample Data Download Complete!")
    print("=" * 60)
    print()
    print("Summary:")
    print(f"  Text: {text_stats['size_mb']:.2f} MB, {text_stats['estimated_tokens']:,} tokens")
    print(f"  Bash: {bash_stats['size_kb']:.2f} KB, {bash_stats['num_scripts']} scripts")
    print()
    print("Next steps:")
    print("  1. Tokenize: python scripts/prepare_sample_data.py")
    print("  2. Train: python scripts/train_sample.py")
    print()
    print("For production training with full datasets (5GB+):")
    print("  python scripts/download_bookcorpus.py")
    print("  python scripts/download_bash_corpus.py")


if __name__ == "__main__":
    main()
