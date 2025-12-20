"""
Download bash scripts from GitHub and other sources.

This script collects a larger corpus of real bash scripts for training:
1. GitHub search API for .sh files
2. Common repositories with quality bash scripts
3. Filters and cleans the scripts

Requirements:
    pip install requests tqdm
"""

import sys
from pathlib import Path
import json
import time
from typing import List, Dict
import requests
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def search_github_bash_scripts(
    query: str = "language:shell",
    max_files: int = 100,
    min_size: int = 50,
    max_size: int = 5000
) -> List[Dict]:
    """
    Search GitHub for bash scripts using the code search API.

    Note: GitHub API has rate limits (60 requests/hour without auth, 5000 with auth).
    Set GITHUB_TOKEN environment variable for authenticated requests.

    Args:
        query: GitHub search query
        max_files: Maximum number of files to download
        min_size: Minimum file size in bytes
        max_size: Maximum file size in bytes

    Returns:
        List of dictionaries with file information
    """
    print(f"Searching GitHub for bash scripts...")
    print(f"  Query: {query}")
    print(f"  Max files: {max_files}")

    # GitHub API endpoints
    search_url = "https://api.github.com/search/code"

    # Headers (add token if available)
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }

    import os
    if 'GITHUB_TOKEN' in os.environ:
        headers['Authorization'] = f"token {os.environ['GITHUB_TOKEN']}"
        print("  Using authenticated GitHub API")
    else:
        print("  Using unauthenticated GitHub API (rate limited)")
        print("  Set GITHUB_TOKEN environment variable for higher limits")

    scripts = []
    page = 1
    per_page = 30  # GitHub max is 100

    while len(scripts) < max_files:
        # Search request
        params = {
            'q': query,
            'per_page': per_page,
            'page': page
        }

        try:
            response = requests.get(search_url, headers=headers, params=params)

            if response.status_code == 403:
                print("\n  Rate limit exceeded. Waiting 60 seconds...")
                time.sleep(60)
                continue

            if response.status_code != 200:
                print(f"\n  Error: {response.status_code}")
                break

            data = response.json()
            items = data.get('items', [])

            if not items:
                break

            # Process items
            for item in items:
                # Get file size
                size = item.get('size', 0)

                # Filter by size
                if size < min_size or size > max_size:
                    continue

                # Get download URL
                html_url = item.get('html_url', '')
                raw_url = html_url.replace('github.com', 'raw.githubusercontent.com')
                raw_url = raw_url.replace('/blob/', '/')

                scripts.append({
                    'name': item.get('name'),
                    'path': item.get('path'),
                    'url': raw_url,
                    'size': size,
                    'repo': item.get('repository', {}).get('full_name')
                })

                if len(scripts) >= max_files:
                    break

            print(f"  Found {len(scripts)} scripts so far...")
            page += 1
            time.sleep(2)  # Be nice to GitHub

        except Exception as e:
            print(f"\n  Error during search: {e}")
            break

    print(f"\nFound {len(scripts)} bash scripts")
    return scripts


def download_scripts(script_infos: List[Dict], max_retries: int = 3) -> List[str]:
    """
    Download bash scripts from URLs.

    Args:
        script_infos: List of script info dicts with 'url' key
        max_retries: Maximum download retries

    Returns:
        List of script contents
    """
    print(f"\nDownloading {len(script_infos)} scripts...")

    scripts = []

    for info in tqdm(script_infos):
        url = info['url']

        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    content = response.text

                    # Basic validation
                    if len(content.strip()) > 0:
                        scripts.append(content)
                    break

                elif response.status_code == 404:
                    break  # Don't retry 404s

                else:
                    time.sleep(1)

            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"\n  Failed to download {url}: {e}")
                time.sleep(1)

        time.sleep(0.5)  # Rate limiting

    print(f"\nSuccessfully downloaded {len(scripts)} scripts")
    return scripts


def get_curated_bash_scripts() -> List[str]:
    """
    Get curated bash scripts from well-known sources.

    These are high-quality examples from documentation and tutorials.
    """
    print("\nFetching curated bash examples...")

    scripts = []

    # Common bash patterns and examples
    curated_examples = [
        # File processing
        """#!/bin/bash
# Process all text files in a directory
for file in *.txt; do
    if [ -f "$file" ]; then
        echo "Processing: $file"
        wc -l "$file"
    fi
done
""",
        # Backup script
        """#!/bin/bash
# Simple backup script
BACKUP_DIR="/backup"
SOURCE_DIR="/data"
DATE=$(date +%Y%m%d)

if [ ! -d "$BACKUP_DIR" ]; then
    mkdir -p "$BACKUP_DIR"
fi

tar -czf "$BACKUP_DIR/backup_$DATE.tar.gz" "$SOURCE_DIR"
echo "Backup completed: backup_$DATE.tar.gz"
""",
        # System monitoring
        """#!/bin/bash
# Check disk space and alert if low
THRESHOLD=90

df -H | grep -vE '^Filesystem|tmpfs|cdrom' | awk '{ print $5 " " $1 }' | while read output; do
    usage=$(echo $output | awk '{ print $1}' | cut -d'%' -f1)
    partition=$(echo $output | awk '{ print $2 }')

    if [ $usage -ge $THRESHOLD ]; then
        echo "Warning: $partition is ${usage}% full"
    fi
done
""",
        # User management
        """#!/bin/bash
# Create multiple users from a file
while IFS=, read -r username fullname email; do
    echo "Creating user: $username"
    useradd -m -c "$fullname" "$username"
    echo "User $username created"
done < users.csv
""",
        # Log analysis
        """#!/bin/bash
# Analyze web server logs
LOG_FILE="/var/log/apache2/access.log"

echo "Top 10 IP addresses:"
awk '{print $1}' "$LOG_FILE" | sort | uniq -c | sort -rn | head -10

echo -e "\nTop 10 requested pages:"
awk '{print $7}' "$LOG_FILE" | sort | uniq -c | sort -rn | head -10
""",
        # Service manager
        """#!/bin/bash
# Simple service manager
SERVICE=$1
ACTION=$2

case $ACTION in
    start)
        echo "Starting $SERVICE..."
        systemctl start "$SERVICE"
        ;;
    stop)
        echo "Stopping $SERVICE..."
        systemctl stop "$SERVICE"
        ;;
    restart)
        echo "Restarting $SERVICE..."
        systemctl restart "$SERVICE"
        ;;
    status)
        systemctl status "$SERVICE"
        ;;
    *)
        echo "Usage: $0 <service> {start|stop|restart|status}"
        exit 1
        ;;
esac
""",
        # File cleanup
        """#!/bin/bash
# Clean old files from directory
DAYS=30
TARGET_DIR="/tmp"

find "$TARGET_DIR" -type f -mtime +$DAYS -print -delete

echo "Cleaned files older than $DAYS days from $TARGET_DIR"
""",
        # Database backup
        """#!/bin/bash
# MySQL database backup
DB_USER="root"
DB_PASS="password"
DB_NAME="mydb"
BACKUP_PATH="/backup/mysql"
DATE=$(date +%Y%m%d_%H%M%S)

mysqldump -u"$DB_USER" -p"$DB_PATH" "$DB_NAME" > "$BACKUP_PATH/${DB_NAME}_${DATE}.sql"

# Keep only last 7 backups
cd "$BACKUP_PATH"
ls -t ${DB_NAME}_*.sql | tail -n +8 | xargs rm -f

echo "Database backup completed"
""",
        # Network test
        """#!/bin/bash
# Test connectivity to multiple hosts
HOSTS=("google.com" "github.com" "stackoverflow.com")

for host in "${HOSTS[@]}"; do
    if ping -c 1 "$host" &> /dev/null; then
        echo "✓ $host is reachable"
    else
        echo "✗ $host is not reachable"
    fi
done
""",
        # Report generator
        """#!/bin/bash
# Generate system report
REPORT="system_report_$(date +%Y%m%d).txt"

{
    echo "System Report - $(date)"
    echo "================================"
    echo
    echo "Hostname: $(hostname)"
    echo "Uptime: $(uptime)"
    echo
    echo "CPU Info:"
    lscpu | grep -E 'Model name|CPU\(s\)'
    echo
    echo "Memory:"
    free -h
    echo
    echo "Disk Usage:"
    df -h
    echo
    echo "Top Processes:"
    ps aux --sort=-%mem | head -10
} > "$REPORT"

echo "Report generated: $REPORT"
"""
    ]

    scripts.extend(curated_examples)

    print(f"  Added {len(curated_examples)} curated examples")

    return scripts


def filter_and_clean_scripts(scripts: List[str]) -> List[str]:
    """
    Filter and clean bash scripts.

    Removes:
    - Empty scripts
    - Very short scripts (< 20 chars)
    - Scripts with encoding errors
    - Duplicate scripts
    """
    print("\nFiltering and cleaning scripts...")

    cleaned = []
    seen = set()

    for script in scripts:
        # Strip whitespace
        script = script.strip()

        # Skip if empty or too short
        if len(script) < 20:
            continue

        # Skip duplicates
        if script in seen:
            continue

        # Basic validation - should look like a script
        # Accept if it has bash-like content
        if any(keyword in script.lower() for keyword in ['bash', 'echo', 'if', 'for', 'while', 'function']):
            cleaned.append(script)
            seen.add(script)

    print(f"  Kept {len(cleaned)} scripts after filtering")

    return cleaned


def save_dataset(scripts: List[str], output_dir: Path):
    """Save scripts to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Individual files
    scripts_dir = output_dir / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    for i, script in enumerate(scripts, 1):
        script_file = scripts_dir / f"script_{i:04d}.sh"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(script)

    # Combined text
    combined_file = output_dir / "bash_scripts.txt"
    with open(combined_file, 'w', encoding='utf-8') as f:
        for script in scripts:
            f.write(script)
            f.write("\n\n" + "="*60 + "\n\n")

    # JSON
    json_file = output_dir / "bash_scripts.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'scripts': scripts,
            'count': len(scripts)
        }, f, indent=2, ensure_ascii=False)

    # Statistics
    stats_file = output_dir / "stats.json"
    total_chars = sum(len(s) for s in scripts)
    total_lines = sum(s.count('\n') + 1 for s in scripts)

    stats = {
        'num_scripts': len(scripts),
        'total_chars': total_chars,
        'total_lines': total_lines,
        'avg_chars': total_chars / len(scripts) if scripts else 0,
        'avg_lines': total_lines / len(scripts) if scripts else 0
    }

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nDataset saved to {output_dir}")
    print(f"  Scripts: {len(scripts)}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Total lines: {total_lines:,}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Download bash scripts from GitHub')
    parser.add_argument('--max-files', type=int, default=100,
                        help='Maximum number of files to download')
    parser.add_argument('--skip-github', action='store_true',
                        help='Skip GitHub download (use curated only)')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent.parent / 'data_large',
                        help='Output directory')

    args = parser.parse_args()

    print("="*60)
    print("Bash Script Dataset Downloader")
    print("="*60)
    print()

    all_scripts = []

    # Get curated examples (always include these)
    curated = get_curated_bash_scripts()
    all_scripts.extend(curated)

    # Download from GitHub
    if not args.skip_github:
        try:
            # Search queries
            queries = [
                "filename:*.sh language:shell stars:>10",
                "extension:sh language:shell",
            ]

            for query in queries:
                script_infos = search_github_bash_scripts(
                    query=query,
                    max_files=args.max_files // len(queries)
                )

                if script_infos:
                    github_scripts = download_scripts(script_infos)
                    all_scripts.extend(github_scripts)

                # Avoid rate limiting
                time.sleep(5)

        except Exception as e:
            print(f"\nGitHub download failed: {e}")
            print("Continuing with curated examples only...")

    # Filter and clean
    cleaned_scripts = filter_and_clean_scripts(all_scripts)

    # Save
    save_dataset(cleaned_scripts, args.output_dir)

    print("\n" + "="*60)
    print("Download complete!")
    print("="*60)


if __name__ == '__main__':
    main()
