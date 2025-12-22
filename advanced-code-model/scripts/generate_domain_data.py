#!/usr/bin/env python3
"""
Domain Training Data Generator

This script orchestrates the generation of training data from various domain modules.
It supports generating data for specific domains or all domains at once.

Usage:
    # Generate all domains
    python generate_domain_data.py --all

    # Generate specific domains
    python generate_domain_data.py --domains sql aws terraform

    # List available domains
    python generate_domain_data.py --list

    # Generate and save to specific output
    python generate_domain_data.py --domains ml data --output data/processed/domain_train.npy
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.domains import (
    get_domain,
    list_domains,
    BaseDomain,
    DomainExample,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate domain-specific training data for code model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate all domain data
    python generate_domain_data.py --all

    # Generate specific domains
    python generate_domain_data.py --domains sql aws terraform mermaid

    # Generate SQL-specific flavors
    python generate_domain_data.py --domains postgres mysql snowflake

    # Generate cloud providers only
    python generate_domain_data.py --domains aws gcp azure

    # List available domains
    python generate_domain_data.py --list

    # Custom output path
    python generate_domain_data.py --all --output data/processed/custom_domain.npy

    # Dry run (show what would be generated)
    python generate_domain_data.py --domains ml data --dry-run
        """
    )

    parser.add_argument(
        "--domains", "-d",
        nargs="+",
        help="Domains to generate data for (space-separated)"
    )

    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Generate data for all available domains"
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available domains and exit"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for training data (default: data/processed/domain_train.npy)"
    )

    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)"
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum sequence length (default: 1024)"
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer (default: data/tokenizer/tokenizer.json)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without actually generating"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show detailed statistics for each domain"
    )

    return parser.parse_args()


def list_all_domains():
    """Print all available domains with descriptions."""
    print("\n" + "=" * 60)
    print("Available Domains")
    print("=" * 60)

    # Group domains by category
    categories = {
        "SQL & Databases": ["sql", "postgres", "mysql", "oracle", "sqlite", "redshift", "snowflake", "sparksql"],
        "Machine Learning": ["ml", "ai", "machinelearning"],
        "Data Engineering": ["data", "bigdata", "pyspark", "glue", "etl"],
        "Cloud Providers": ["aws", "gcp", "azure"],
        "DevOps & IaC": ["devops", "terraform", "cloudformation", "kubernetes", "docker", "cicd"],
        "Diagrams": ["diagrams", "mermaid", "plantuml", "drawio"],
        "Web & Mobile": ["web", "mobile", "frontend", "backend"],
    }

    for category, domains in categories.items():
        print(f"\n{category}:")
        for domain_name in domains:
            try:
                domain = get_domain(domain_name)
                examples = domain.get_examples()
                print(f"  {domain_name:20} - {domain.get_description()} ({len(examples)} examples)")
            except Exception as e:
                print(f"  {domain_name:20} - (error: {e})")

    print("\n" + "=" * 60)
    print(f"Total domains available: {len(list_domains())}")
    print("=" * 60 + "\n")


def get_domain_examples(domain_names: List[str], verbose: bool = False) -> List[DomainExample]:
    """Get all examples from specified domains."""
    all_examples = []

    for name in domain_names:
        try:
            domain = get_domain(name)
            examples = domain.get_examples()
            all_examples.extend(examples)

            if verbose:
                print(f"  {name}: {len(examples)} examples")

        except Exception as e:
            print(f"  Warning: Failed to load domain '{name}': {e}")

    return all_examples


def format_example(example: DomainExample) -> str:
    """Format a single example for training."""
    # Format: <domain> <subdomain>\n<prompt>\n<code>
    header = f"### Domain: {example.domain}"
    if example.subdomain:
        header += f" / {example.subdomain}"

    if example.tags:
        header += f"\n### Tags: {', '.join(example.tags)}"

    formatted = f"""{header}
### Difficulty: {example.difficulty}

### Prompt:
{example.prompt}

### Code:
{example.code}

### End
"""
    return formatted


def tokenize_examples(
    examples: List[DomainExample],
    tokenizer_path: Path,
    max_length: int = 1024,
    verbose: bool = False
) -> np.ndarray:
    """Tokenize examples into training sequences."""
    from tokenizers import Tokenizer

    print(f"\nLoading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    all_sequences = []
    skipped = 0

    for i, example in enumerate(examples):
        text = format_example(example)
        encoding = tokenizer.encode(text)
        tokens = encoding.ids

        # Create fixed-length sequences
        for j in range(0, len(tokens), max_length):
            chunk = tokens[j:j + max_length]
            if len(chunk) < max_length:
                # Pad with zeros
                chunk = chunk + [0] * (max_length - len(chunk))
            all_sequences.append(chunk)

        if verbose and (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(examples)} examples...")

    if skipped > 0:
        print(f"  Skipped {skipped} examples (too short)")

    return np.array(all_sequences, dtype=np.int32)


def show_stats(examples: List[DomainExample]):
    """Show detailed statistics about the examples."""
    print("\n" + "=" * 60)
    print("Domain Statistics")
    print("=" * 60)

    # Count by domain
    domain_counts = {}
    subdomain_counts = {}
    difficulty_counts = {"beginner": 0, "intermediate": 0, "advanced": 0}
    tag_counts = {}

    for ex in examples:
        domain_counts[ex.domain] = domain_counts.get(ex.domain, 0) + 1

        if ex.subdomain:
            key = f"{ex.domain}/{ex.subdomain}"
            subdomain_counts[key] = subdomain_counts.get(key, 0) + 1

        difficulty_counts[ex.difficulty] = difficulty_counts.get(ex.difficulty, 0) + 1

        for tag in ex.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    print("\nBy Domain:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"  {domain:20} {count:5} examples")

    print("\nBy Subdomain:")
    for subdomain, count in sorted(subdomain_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {subdomain:30} {count:5} examples")

    print("\nBy Difficulty:")
    for diff, count in difficulty_counts.items():
        print(f"  {diff:15} {count:5} examples")

    print("\nTop Tags:")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {tag:20} {count:5}")

    print("\n" + "=" * 60)


def main():
    args = parse_args()

    # Handle list command
    if args.list:
        list_all_domains()
        return

    # Determine which domains to use
    if args.all:
        # Use main domain names (not aliases) to avoid duplicates
        domain_names = ["sql", "ml", "data", "aws", "gcp", "azure", "devops", "diagrams", "web"]
    elif args.domains:
        domain_names = args.domains
    else:
        print("Error: Please specify --domains or --all")
        print("Use --list to see available domains")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Domain Training Data Generator")
    print("=" * 60)
    print(f"Domains: {', '.join(domain_names)}")
    print(f"Max sequence length: {args.max_length}")
    print(f"Validation split: {args.val_split}")

    # Collect examples
    print("\nCollecting examples...")
    examples = get_domain_examples(domain_names, verbose=args.verbose)
    print(f"Total examples collected: {len(examples)}")

    if len(examples) == 0:
        print("Error: No examples found!")
        sys.exit(1)

    # Show stats if requested
    if args.stats:
        show_stats(examples)

    # Dry run - just show what would be generated
    if args.dry_run:
        print("\n[DRY RUN] Would generate:")
        print(f"  - {len(examples)} examples")
        print(f"  - Estimated sequences: ~{len(examples) * 2}")
        print(f"  - Output: {args.output or 'data/processed/domain_train.npy'}")
        return

    # Set up paths
    data_dir = project_root / "data"
    tokenizer_path = Path(args.tokenizer) if args.tokenizer else data_dir / "tokenizer" / "tokenizer.json"
    output_dir = data_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_train = Path(args.output)
        output_val = output_train.parent / output_train.name.replace("_train", "_val").replace(".npy", "_val.npy")
    else:
        output_train = output_dir / "domain_train.npy"
        output_val = output_dir / "domain_val.npy"

    # Check tokenizer exists
    if not tokenizer_path.exists():
        print(f"\nError: Tokenizer not found at {tokenizer_path}")
        print("Run: python scripts/train_tokenizer.py first")
        sys.exit(1)

    # Tokenize
    print("\nTokenizing examples...")
    sequences = tokenize_examples(examples, tokenizer_path, args.max_length, args.verbose)
    print(f"Generated {len(sequences)} sequences")

    # Shuffle
    np.random.shuffle(sequences)

    # Split into train/val
    split_idx = int(len(sequences) * (1 - args.val_split))
    train_data = sequences[:split_idx]
    val_data = sequences[split_idx:]

    print(f"\nTraining sequences: {len(train_data)}")
    print(f"Validation sequences: {len(val_data)}")

    # Save
    np.save(output_train, train_data)
    np.save(output_val, val_data)

    print(f"\nSaved training data: {output_train}")
    print(f"Saved validation data: {output_val}")

    # Calculate sizes
    train_tokens = train_data.shape[0] * train_data.shape[1]
    val_tokens = val_data.shape[0] * val_data.shape[1]
    train_mb = train_data.nbytes / (1024 * 1024)
    val_mb = val_data.nbytes / (1024 * 1024)

    print(f"\nDataset Statistics:")
    print(f"  Training: {train_data.shape[0]:,} sequences, {train_tokens:,} tokens ({train_mb:.1f} MB)")
    print(f"  Validation: {val_data.shape[0]:,} sequences, {val_tokens:,} tokens ({val_mb:.1f} MB)")

    # Show training command
    print("\n" + "=" * 60)
    print("Next Steps")
    print("=" * 60)
    print(f"""
To train with domain data:

python scripts/train.py \\
    --stage domain \\
    --checkpoint models/code_model_best.pth \\
    --data-file domain_train.npy \\
    --model-size large \\
    --batch-size 2 \\
    --num-epochs 5 \\
    --learning-rate 3e-6 \\
    --use-rmsnorm --use-rope --use-compile --use-amp
""")


if __name__ == "__main__":
    main()
