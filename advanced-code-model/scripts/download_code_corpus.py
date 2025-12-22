#!/usr/bin/env python3
"""
Download code corpus from HuggingFace for code model training.

Uses datasets in Parquet format that work with datasets v4+.

Usage:
    python scripts/download_code_corpus.py
    python scripts/download_code_corpus.py --samples 50000
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import random

from datasets import load_dataset
from tqdm import tqdm


def download_mbpp() -> tuple:
    """Download MBPP - Python programming problems."""
    print("\n" + "=" * 60)
    print("Downloading MBPP Dataset")
    print("=" * 60)

    all_code = []

    try:
        dataset = load_dataset("mbpp", split="train")

        for item in tqdm(dataset, desc="Processing MBPP"):
            code = item.get("code", "")
            prompt = item.get("text", "")
            tests = item.get("test_list", [])

            if len(code) > 50:
                test_str = "\n".join(tests[:3]) if tests else ""
                full_code = f"# Task: {prompt}\n\n{code}\n\n# Tests:\n{test_str}"
                all_code.append({
                    "code": full_code,
                    "language": "python",
                    "size": len(full_code),
                })

        print(f"  ✓ MBPP: {len(all_code):,} Python samples")

    except Exception as e:
        print(f"  ✗ Error: {e}")

    return all_code, {"samples": len(all_code)}


def download_humaneval() -> tuple:
    """Download HumanEval - Python coding problems."""
    print("\n" + "=" * 60)
    print("Downloading HumanEval Dataset")
    print("=" * 60)

    all_code = []

    try:
        dataset = load_dataset("openai_humaneval", split="test")

        for item in tqdm(dataset, desc="Processing HumanEval"):
            prompt = item.get("prompt", "")
            solution = item.get("canonical_solution", "")
            test = item.get("test", "")
            code = f"{prompt}{solution}\n\n{test}"

            if len(code) > 50:
                all_code.append({
                    "code": code,
                    "language": "python",
                    "size": len(code),
                })

        print(f"  ✓ HumanEval: {len(all_code):,} Python samples")

    except Exception as e:
        print(f"  ✗ Error: {e}")

    return all_code, {"samples": len(all_code)}


def download_python_code_instructions(samples: int = 20000) -> tuple:
    """Download iamtarun/python_code_instructions_18k_alpaca."""
    print("\n" + "=" * 60)
    print("Downloading Python Code Instructions Dataset")
    print("=" * 60)

    all_code = []

    try:
        dataset = load_dataset(
            "iamtarun/python_code_instructions_18k_alpaca",
            split="train",
        )

        for item in tqdm(dataset, desc="Processing"):
            instruction = item.get("instruction", "")
            output = item.get("output", "")

            if len(output) > 50 and len(output) < 8000:
                code = f"# Instruction: {instruction}\n\n{output}"
                all_code.append({
                    "code": code,
                    "language": "python",
                    "size": len(code),
                })

            if len(all_code) >= samples:
                break

        print(f"  ✓ Python Code Instructions: {len(all_code):,} samples")

    except Exception as e:
        print(f"  ✗ Error: {e}")

    return all_code, {"samples": len(all_code)}


def download_code_alpaca(samples: int = 20000) -> tuple:
    """Download sahil2801/CodeAlpaca-20k."""
    print("\n" + "=" * 60)
    print("Downloading CodeAlpaca Dataset")
    print("=" * 60)

    all_code = []

    try:
        dataset = load_dataset(
            "sahil2801/CodeAlpaca-20k",
            split="train",
        )

        for item in tqdm(dataset, desc="Processing"):
            instruction = item.get("instruction", "")
            output = item.get("output", "")
            inp = item.get("input", "")

            if len(output) > 50 and len(output) < 8000:
                if inp:
                    code = f"# Instruction: {instruction}\n# Input: {inp}\n\n{output}"
                else:
                    code = f"# Instruction: {instruction}\n\n{output}"

                all_code.append({
                    "code": code,
                    "language": "python",  # Mixed but mostly Python
                    "size": len(code),
                })

            if len(all_code) >= samples:
                break

        print(f"  ✓ CodeAlpaca: {len(all_code):,} samples")

    except Exception as e:
        print(f"  ✗ Error: {e}")

    return all_code, {"samples": len(all_code)}


def download_evol_instruct_code(samples: int = 50000) -> tuple:
    """Download nickrosh/Evol-Instruct-Code-80k-v1."""
    print("\n" + "=" * 60)
    print("Downloading Evol-Instruct-Code Dataset")
    print("=" * 60)

    all_code = []

    try:
        dataset = load_dataset(
            "nickrosh/Evol-Instruct-Code-80k-v1",
            split="train",
        )

        for item in tqdm(dataset, desc="Processing"):
            instruction = item.get("instruction", "")
            output = item.get("output", "")

            if len(output) > 100 and len(output) < 8000:
                code = f"# Task: {instruction[:500]}\n\n{output}"
                all_code.append({
                    "code": code,
                    "language": "mixed",
                    "size": len(code),
                })

            if len(all_code) >= samples:
                break

        print(f"  ✓ Evol-Instruct-Code: {len(all_code):,} samples")

    except Exception as e:
        print(f"  ✗ Error: {e}")

    return all_code, {"samples": len(all_code)}


def download_code_exercises(samples: int = 30000) -> tuple:
    """Download jinaai/code_exercises."""
    print("\n" + "=" * 60)
    print("Downloading Code Exercises Dataset")
    print("=" * 60)

    all_code = []

    try:
        dataset = load_dataset(
            "jinaai/code_exercises",
            split="train",
            streaming=True,
        )

        pbar = tqdm(total=samples, desc="Processing")

        for item in dataset:
            problem = item.get("problem", "")
            solution = item.get("solution", "")

            if len(solution) > 50 and len(solution) < 8000:
                code = f"# Problem:\n# {problem[:500]}\n\n{solution}"
                all_code.append({
                    "code": code,
                    "language": "python",
                    "size": len(code),
                })
                pbar.update(1)

            if len(all_code) >= samples:
                break

        pbar.close()
        print(f"  ✓ Code Exercises: {len(all_code):,} samples")

    except Exception as e:
        print(f"  ✗ Error: {e}")

    return all_code, {"samples": len(all_code)}


def download_glaive_code(samples: int = 50000) -> tuple:
    """Download glaiveai/glaive-code-assistant."""
    print("\n" + "=" * 60)
    print("Downloading Glaive Code Assistant Dataset")
    print("=" * 60)

    all_code = []

    try:
        dataset = load_dataset(
            "glaiveai/glaive-code-assistant",
            split="train",
            streaming=True,
        )

        pbar = tqdm(total=samples, desc="Processing")

        for item in dataset:
            # Parse conversation format
            messages = item.get("conversation", "") or item.get("text", "")

            if len(messages) > 100 and len(messages) < 10000:
                all_code.append({
                    "code": messages,
                    "language": "mixed",
                    "size": len(messages),
                })
                pbar.update(1)

            if len(all_code) >= samples:
                break

        pbar.close()
        print(f"  ✓ Glaive Code: {len(all_code):,} samples")

    except Exception as e:
        print(f"  ✗ Error: {e}")

    return all_code, {"samples": len(all_code)}


def generate_bash_examples(num_examples: int = 5000) -> tuple:
    """Generate synthetic bash script examples."""
    print("\n" + "=" * 60)
    print("Generating Synthetic Bash Examples")
    print("=" * 60)

    templates = [
        # File operations
        '''#!/bin/bash
# {desc}
set -euo pipefail

{var}="${{1:-{default}}}"

if [[ -f "${var}" ]]; then
    {action}
    echo "Done processing ${var}"
else
    echo "Error: File not found: ${var}" >&2
    exit 1
fi''',

        # Directory operations
        '''#!/bin/bash
# {desc}
set -euo pipefail

SOURCE_DIR="${{1:-.}}"
DEST_DIR="${{2:-/tmp/backup}}"

mkdir -p "$DEST_DIR"

for file in "$SOURCE_DIR"/{pattern}; do
    if [[ -f "$file" ]]; then
        {action}
    fi
done

echo "Processed $(ls "$DEST_DIR" | wc -l) files"''',

        # Service management
        '''#!/bin/bash
# {desc}

SERVICE_NAME="${{1:-{service}}}"

start_service() {{
    echo "Starting $SERVICE_NAME..."
    {start_cmd}
}}

stop_service() {{
    echo "Stopping $SERVICE_NAME..."
    {stop_cmd}
}}

status_service() {{
    {status_cmd}
}}

case "${{2:-status}}" in
    start)  start_service ;;
    stop)   stop_service ;;
    status) status_service ;;
    restart) stop_service && start_service ;;
    *)      echo "Usage: $0 <service> {{start|stop|status|restart}}" ;;
esac''',

        # Log processing
        '''#!/bin/bash
# {desc}
set -euo pipefail

LOG_FILE="${{1:-/var/log/{logfile}}}"
PATTERN="${{2:-{pattern}}}"

if [[ ! -f "$LOG_FILE" ]]; then
    echo "Log file not found: $LOG_FILE" >&2
    exit 1
fi

echo "=== Log Analysis: $LOG_FILE ==="
echo "Pattern: $PATTERN"
echo

MATCHES=$(grep -c "$PATTERN" "$LOG_FILE" 2>/dev/null || echo 0)
echo "Total matches: $MATCHES"

if [[ $MATCHES -gt 0 ]]; then
    echo
    echo "Recent matches:"
    grep "$PATTERN" "$LOG_FILE" | tail -10
fi''',

        # Git automation
        '''#!/bin/bash
# {desc}
set -euo pipefail

BRANCH="${{1:-main}}"
MESSAGE="${{2:-{message}}}"

# Check for changes
if [[ -z $(git status -s) ]]; then
    echo "No changes to commit"
    exit 0
fi

# Stage and commit
git add -A
git commit -m "$MESSAGE"

# Push if remote exists
if git remote | grep -q origin; then
    git push origin "$BRANCH"
    echo "Changes pushed to origin/$BRANCH"
else
    echo "Committed locally (no remote configured)"
fi''',

        # Docker operations
        '''#!/bin/bash
# {desc}
set -euo pipefail

IMAGE_NAME="${{1:-{image}}}"
CONTAINER_NAME="${{2:-{container}}}"

build_image() {{
    echo "Building image: $IMAGE_NAME"
    docker build -t "$IMAGE_NAME" .
}}

run_container() {{
    echo "Running container: $CONTAINER_NAME"
    docker run -d --name "$CONTAINER_NAME" {docker_opts} "$IMAGE_NAME"
}}

cleanup() {{
    echo "Cleaning up..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
}}

case "${{3:-build}}" in
    build) build_image ;;
    run)   run_container ;;
    clean) cleanup ;;
    *)     echo "Usage: $0 <image> <container> {{build|run|clean}}" ;;
esac''',

        # System monitoring
        '''#!/bin/bash
# {desc}

THRESHOLD="${{1:-{threshold}}}"
ALERT_EMAIL="${{2:-admin@example.com}}"

check_disk() {{
    USAGE=$(df -h / | awk 'NR==2 {{print $5}}' | tr -d '%')
    if [[ $USAGE -gt $THRESHOLD ]]; then
        echo "ALERT: Disk usage is $USAGE% (threshold: $THRESHOLD%)"
        return 1
    fi
    echo "Disk usage OK: $USAGE%"
}}

check_memory() {{
    FREE=$(free | awk '/^Mem:/ {{printf "%.0f", $4/$2 * 100}}')
    if [[ $FREE -lt 10 ]]; then
        echo "ALERT: Low memory - only $FREE% free"
        return 1
    fi
    echo "Memory OK: $FREE% free"
}}

check_load() {{
    LOAD=$(uptime | awk -F'load average:' '{{print $2}}' | cut -d, -f1 | tr -d ' ')
    CORES=$(nproc)
    if (( $(echo "$LOAD > $CORES" | bc -l) )); then
        echo "ALERT: High load - $LOAD (cores: $CORES)"
        return 1
    fi
    echo "Load OK: $LOAD"
}}

echo "=== System Health Check ==="
check_disk
check_memory
check_load
echo "=== Check Complete ==="''',

        # Backup script
        '''#!/bin/bash
# {desc}
set -euo pipefail

SOURCE="${{1:-{source}}}"
BACKUP_DIR="${{2:-/backup}}"
RETENTION_DAYS="${{3:-7}}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/backup_$TIMESTAMP.tar.gz"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Create backup
echo "Creating backup: $BACKUP_FILE"
tar -czf "$BACKUP_FILE" -C "$(dirname "$SOURCE")" "$(basename "$SOURCE")"

# Verify backup
if tar -tzf "$BACKUP_FILE" >/dev/null 2>&1; then
    SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
    echo "Backup created successfully: $SIZE"
else
    echo "Backup verification failed!" >&2
    exit 1
fi

# Cleanup old backups
echo "Cleaning up backups older than $RETENTION_DAYS days..."
find "$BACKUP_DIR" -name "backup_*.tar.gz" -mtime +$RETENTION_DAYS -delete

echo "Backup complete"''',
    ]

    variables = {
        "desc": ["Backup files", "Process logs", "Deploy application", "Check system health",
                 "Manage Docker containers", "Git automation", "Service manager", "File processor"],
        "var": ["FILE", "INPUT", "TARGET", "SOURCE"],
        "default": ["/tmp/data.txt", "input.txt", "$HOME/data", "/var/log/app.log"],
        "action": ["cat \"$FILE\"", "wc -l \"$FILE\"", "cp \"$FILE\" /backup/", "gzip \"$FILE\""],
        "pattern": ["*.txt", "*.log", "*.sh", "*.py", "*.json"],
        "service": ["nginx", "postgres", "redis", "app"],
        "start_cmd": ["systemctl start $SERVICE_NAME", "docker start $SERVICE_NAME"],
        "stop_cmd": ["systemctl stop $SERVICE_NAME", "docker stop $SERVICE_NAME"],
        "status_cmd": ["systemctl status $SERVICE_NAME", "docker ps | grep $SERVICE_NAME"],
        "logfile": ["syslog", "app.log", "access.log", "error.log"],
        "message": ["Auto commit", "Update files", "Fix bugs", "Add feature"],
        "image": ["myapp", "webapp", "api-server"],
        "container": ["myapp-container", "web-1", "api-1"],
        "docker_opts": ["-p 8080:80", "-p 3000:3000 -e NODE_ENV=production", "--rm"],
        "threshold": ["80", "90", "75"],
        "source": ["/var/www", "/home/user/data", "/opt/app"],
    }

    examples = []

    for i in range(num_examples):
        template = random.choice(templates)
        script = template

        for var, options in variables.items():
            placeholder = "{" + var + "}"
            while placeholder in script:
                script = script.replace(placeholder, random.choice(options), 1)

        examples.append({
            "code": script,
            "language": "bash",
            "size": len(script),
        })

    print(f"  ✓ Generated {len(examples):,} bash examples")
    return examples, {"samples": len(examples)}


def save_code_corpus(
    code_samples: List[Dict],
    output_dir: Path,
    stats: Dict,
) -> Path:
    """Save code corpus to text file for training."""

    output_dir.mkdir(parents=True, exist_ok=True)
    random.shuffle(code_samples)

    corpus_file = output_dir / "code_corpus.txt"

    print(f"\nSaving corpus to {corpus_file}...")

    with open(corpus_file, "w", encoding="utf-8") as f:
        for sample in tqdm(code_samples, desc="Writing"):
            lang = sample.get("language", "unknown")
            code = sample.get("code", "")

            f.write(f"### Language: {lang}\n")
            f.write(f"### Code:\n")
            f.write(code)
            f.write("\n### End\n\n")

    stats_file = output_dir / "code_corpus_stats.json"
    with open(stats_file, "w") as f:
        json.dump({
            "total_samples": len(code_samples),
            "by_source": stats,
        }, f, indent=2)

    file_size = corpus_file.stat().st_size / (1024 * 1024)
    print(f"\n✓ Saved {len(code_samples):,} samples ({file_size:.1f} MB)")

    return corpus_file


def main():
    parser = argparse.ArgumentParser(description="Download code corpus")
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--max-samples", type=int, default=100000,
                        help="Max samples per dataset (default: 100000)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Code Corpus Downloader")
    print("=" * 60)
    print(f"Output: {args.output}")

    all_code = []
    all_stats = {}

    # 1. MBPP
    code, stats = download_mbpp()
    all_code.extend(code)
    all_stats["mbpp"] = stats

    # 2. HumanEval
    code, stats = download_humaneval()
    all_code.extend(code)
    all_stats["humaneval"] = stats

    # 3. Python Code Instructions
    code, stats = download_python_code_instructions(min(20000, args.max_samples))
    all_code.extend(code)
    all_stats["python_code_instructions"] = stats

    # 4. CodeAlpaca
    code, stats = download_code_alpaca(min(20000, args.max_samples))
    all_code.extend(code)
    all_stats["code_alpaca"] = stats

    # 5. Evol-Instruct-Code
    code, stats = download_evol_instruct_code(min(50000, args.max_samples))
    all_code.extend(code)
    all_stats["evol_instruct_code"] = stats

    # 6. Glaive Code
    code, stats = download_glaive_code(min(30000, args.max_samples))
    all_code.extend(code)
    all_stats["glaive_code"] = stats

    # 7. Code Exercises
    code, stats = download_code_exercises(min(30000, args.max_samples))
    all_code.extend(code)
    all_stats["code_exercises"] = stats

    # 8. Synthetic Bash
    code, stats = generate_bash_examples(5000)
    all_code.extend(code)
    all_stats["synthetic_bash"] = stats

    if not all_code:
        print("\n✗ No code downloaded!")
        return

    # Save corpus
    corpus_file = save_code_corpus(all_code, args.output, all_stats)

    # Print summary
    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)

    total_samples = len(all_code)
    total_chars = sum(s.get("size", 0) for s in all_code)

    print(f"Total samples: {total_samples:,}")
    print(f"Total size: {total_chars/1e6:.1f}M chars (~{total_chars/4/1e6:.1f}M tokens)")

    lang_counts = {}
    for sample in all_code:
        lang = sample.get("language", "unknown")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    print("\nBy language:")
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"  {lang:15} {count:,} samples")

    print("\nNext steps:")
    print("  1. python scripts/prepare_code_data.py")
    print("  2. python scripts/train.py --stage code --data-file code_train_large.npy")


if __name__ == "__main__":
    main()
