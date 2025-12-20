"""
Download and curate large bash script corpus for code fine-tuning.

Collects bash scripts from multiple sources:
1. GitHub repositories (popular DevOps/SysAdmin repos)
2. StackOverflow bash solutions
3. Curated production scripts

Target size: 500MB+ (10,000+ scripts)

Usage:
    python scripts/download_bash_corpus.py --output data/bash_scripts
"""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple

import requests
from tqdm import tqdm


# Popular GitHub repositories with quality bash scripts
GITHUB_REPOS = [
    # DevOps & Infrastructure
    "hashicorp/terraform",
    "ansible/ansible",
    "kubernetes/kubernetes",
    "docker/docker-ce",
    "helm/helm",

    # System administration
    "ohmyzsh/ohmyzsh",
    "nvm-sh/nvm",
    "rbenv/rbenv",
    "pyenv/pyenv",

    # CI/CD
    "actions/runner",
    "jenkins-x/jx",
    "drone/drone",

    # Utilities
    "dylanaraps/pure-bash-bible",
    "awesome-lists/awesome-bash",
    "alexanderepstein/Bash-Snippets",
]


def clone_github_repo(repo: str, output_dir: Path) -> Path:
    """
    Clone a GitHub repository.

    Args:
        repo: Repository name (owner/repo)
        output_dir: Directory to clone into

    Returns:
        Path to cloned repository
    """
    repo_dir = output_dir / repo.replace("/", "_")

    if repo_dir.exists():
        print(f"  ✓ Already cloned: {repo}")
        return repo_dir

    url = f"https://github.com/{repo}.git"

    try:
        print(f"  Cloning {repo}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(repo_dir)],
            check=True,
            capture_output=True,
            timeout=300
        )
        print(f"  ✓ Cloned: {repo}")
        return repo_dir

    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout cloning {repo}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error cloning {repo}: {e}")
        return None


def extract_bash_scripts(repo_dir: Path) -> List[Tuple[Path, str]]:
    """
    Extract bash scripts from a repository.

    Args:
        repo_dir: Path to repository

    Returns:
        List of (file_path, content) tuples
    """
    scripts = []

    # Find all .sh files
    sh_files = list(repo_dir.rglob("*.sh"))

    # Also find scripts with bash shebang
    for file in repo_dir.rglob("*"):
        if file.is_file() and not file.suffix == ".sh":
            try:
                with open(file, "r", encoding="utf-8", errors="ignore") as f:
                    first_line = f.readline()
                    if "#!/bin/bash" in first_line or "#!/usr/bin/env bash" in first_line:
                        sh_files.append(file)
            except:
                pass

    # Read and validate scripts
    for script_path in sh_files:
        try:
            with open(script_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Basic validation
            if len(content) < 50:  # Too short
                continue

            if len(content) > 50000:  # Too long (>50KB)
                continue

            # Must have shebang
            if not content.strip().startswith("#!"):
                continue

            scripts.append((script_path, content))

        except Exception as e:
            continue

    return scripts


def is_valid_bash_script(content: str) -> bool:
    """
    Validate bash script quality.

    Args:
        content: Script content

    Returns:
        True if script meets quality criteria
    """
    lines = content.split("\n")

    # Must have shebang
    if not lines[0].strip().startswith("#!"):
        return False

    # Must have some actual code (not just comments)
    code_lines = [l for l in lines if l.strip() and not l.strip().startswith("#")]
    if len(code_lines) < 5:
        return False

    # Check for common bash patterns (indicates real script)
    patterns = [
        r"\bif\b.*\bthen\b",
        r"\bfor\b.*\bin\b",
        r"\bwhile\b.*\bdo\b",
        r"\bfunction\b",
        r"\$\{.*\}",  # Variable expansion
        r"\becho\b",
        r"\bset\b",
    ]

    pattern_matches = sum(
        1 for pattern in patterns
        if re.search(pattern, content, re.MULTILINE)
    )

    # Should match at least 2 patterns
    if pattern_matches < 2:
        return False

    # No binary data
    try:
        content.encode("ascii", errors="strict")
    except:
        # Contains non-ASCII (might be binary)
        non_ascii_ratio = len([c for c in content if ord(c) > 127]) / len(content)
        if non_ascii_ratio > 0.05:  # More than 5% non-ASCII
            return False

    return True


def deduplicate_scripts(scripts: List[Tuple[Path, str]]) -> List[Tuple[Path, str]]:
    """
    Remove duplicate scripts based on content hash.

    Args:
        scripts: List of (path, content) tuples

    Returns:
        Deduplicated list
    """
    seen_hashes: Set[str] = set()
    unique_scripts = []

    for path, content in scripts:
        # Hash content
        content_hash = hashlib.md5(content.encode()).hexdigest()

        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_scripts.append((path, content))

    return unique_scripts


def download_bash_corpus(output_dir: Path) -> Dict:
    """
    Download comprehensive bash script corpus.

    Args:
        output_dir: Output directory

    Returns:
        Statistics dictionary
    """
    print("=" * 60)
    print("Downloading Bash Script Corpus")
    print("=" * 60)
    print()

    # Create directories
    raw_dir = output_dir / "raw"
    repos_dir = raw_dir / "repos"
    scripts_dir = raw_dir / "scripts"

    repos_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)

    all_scripts = []

    # Clone GitHub repositories
    print(f"Cloning {len(GITHUB_REPOS)} GitHub repositories...")
    print("This may take 15-30 minutes.")
    print()

    for repo in tqdm(GITHUB_REPOS, desc="Cloning repos"):
        repo_dir = clone_github_repo(repo, repos_dir)

        if repo_dir:
            # Extract bash scripts
            scripts = extract_bash_scripts(repo_dir)
            all_scripts.extend(scripts)

    print()
    print(f"✓ Extracted {len(all_scripts):,} raw scripts")
    print()

    # Validate and filter
    print("Validating scripts...")
    valid_scripts = []

    for path, content in tqdm(all_scripts, desc="Validating"):
        if is_valid_bash_script(content):
            valid_scripts.append((path, content))

    print(f"✓ {len(valid_scripts):,} valid scripts")
    print()

    # Deduplicate
    print("Removing duplicates...")
    unique_scripts = deduplicate_scripts(valid_scripts)
    print(f"✓ {len(unique_scripts):,} unique scripts")
    print()

    # Save scripts
    print("Saving scripts...")
    saved_scripts = []

    for idx, (original_path, content) in enumerate(tqdm(unique_scripts, desc="Saving")):
        script_file = scripts_dir / f"script_{idx+1:06d}.sh"

        with open(script_file, "w", encoding="utf-8") as f:
            f.write(content)

        saved_scripts.append({
            "id": idx + 1,
            "file": script_file.name,
            "original_path": str(original_path),
            "size": len(content),
            "lines": content.count("\n") + 1,
        })

    print()
    print(f"✓ Saved {len(saved_scripts):,} scripts")
    print()

    # Calculate statistics
    total_size = sum(s["size"] for s in saved_scripts)
    total_lines = sum(s["lines"] for s in saved_scripts)
    total_words = sum(len(content.split()) for _, content in unique_scripts)

    stats = {
        "num_scripts": len(saved_scripts),
        "total_characters": total_size,
        "total_lines": total_lines,
        "total_words": total_words,
        "estimated_tokens": int(total_words * 1.3),  # BPE approximation
        "size_mb": total_size / (1024 ** 2),
        "avg_lines_per_script": total_lines / len(saved_scripts),
        "avg_size_bytes": total_size / len(saved_scripts),
    }

    # Save metadata
    metadata = {
        "stats": stats,
        "scripts": saved_scripts,
        "sources": {
            "github_repos": GITHUB_REPOS,
        }
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save stats
    stats_file = output_dir / "stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    # Print statistics
    print("Corpus Statistics:")
    print(f"  Scripts: {stats['num_scripts']:,}")
    print(f"  Total lines: {stats['total_lines']:,}")
    print(f"  Total words: {stats['total_words']:,}")
    print(f"  Estimated tokens: {stats['estimated_tokens']:,}")
    print(f"  Size: {stats['size_mb']:.2f} MB")
    print(f"  Avg lines/script: {stats['avg_lines_per_script']:.0f}")
    print()
    print(f"✓ Metadata saved to {metadata_file}")

    return stats


def verify_corpus(output_dir: Path) -> bool:
    """Verify downloaded corpus."""
    print()
    print("Verifying corpus...")

    scripts_dir = output_dir / "raw" / "scripts"
    if not scripts_dir.exists():
        print("✗ Scripts directory not found")
        return False

    script_files = list(scripts_dir.glob("script_*.sh"))
    if len(script_files) < 1000:
        print(f"⚠ Warning: Only {len(script_files)} scripts found (expected 10,000+)")
        return False

    stats_file = output_dir / "stats.json"
    if not stats_file.exists():
        print("✗ Statistics file not found")
        return False

    with open(stats_file) as f:
        stats = json.load(f)

    if stats["size_mb"] < 100:
        print(f"⚠ Warning: Corpus size ({stats['size_mb']:.2f} MB) is smaller than expected (500MB+)")

    print(f"✓ Found {len(script_files):,} scripts")
    print(f"✓ Total size: {stats['size_mb']:.2f} MB")
    print()
    print("✓ Verification passed!")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download bash script corpus for code fine-tuning"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/bash_scripts"),
        help="Output directory (default: data/bash_scripts)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing download"
    )

    args = parser.parse_args()

    if args.verify_only:
        success = verify_corpus(args.output)
        exit(0 if success else 1)

    # Download corpus
    stats = download_bash_corpus(args.output)

    # Verify
    success = verify_corpus(args.output)

    if success:
        print()
        print("=" * 60)
        print("Bash corpus download complete!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  1. Process: python scripts/prepare_bash_corpus.py")
        print("  2. Fine-tune: python scripts/train_finetune.py")
        exit(0)
    else:
        print()
        print("✗ Verification failed.")
        exit(1)


if __name__ == "__main__":
    main()
