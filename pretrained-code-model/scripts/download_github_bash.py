"""
Download 10,000+ bash scripts from GitHub.

Uses GitHub Code Search API to find high-quality bash scripts.
Set GITHUB_TOKEN environment variable for higher rate limits (5000/hour vs 60/hour).

Sources:
- Popular repositories with .sh files
- Repositories with "bash", "shell", "scripts" topics
- Files with high stars/forks
"""

import sys
from pathlib import Path
import json
import time
import requests
from tqdm import tqdm
import os

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class GitHubBashDownloader:
    """Download bash scripts from GitHub."""

    def __init__(self, token=None):
        self.session = requests.Session()
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "BashScriptDownloader/1.0"
        }

        if token:
            self.headers["Authorization"] = f"token {token}"
            self.rate_limit = 5000
            print("✓ Using authenticated GitHub API (5000 requests/hour)")
        else:
            self.rate_limit = 60
            print("⚠ Using unauthenticated API (60 requests/hour)")
            print("  Set GITHUB_TOKEN for higher limits")

        self.scripts = []
        self.seen_urls = set()

    def search_code(self, query, max_results=100):
        """Search GitHub for code matching query."""
        print(f"\nSearching: {query}")

        url = "https://api.github.com/search/code"
        scripts_found = []

        page = 1
        per_page = 30  # GitHub max is 100, but 30 is safer

        while len(scripts_found) < max_results:
            params = {
                'q': query,
                'per_page': per_page,
                'page': page,
                'sort': 'indexed'  # Get recently indexed (more variety)
            }

            try:
                response = self.session.get(url, headers=self.headers, params=params, timeout=30)

                # Check rate limit
                if response.status_code == 403:
                    reset_time = int(response.headers.get('X-RateLimit-Reset', time.time() + 60))
                    wait_time = max(reset_time - time.time(), 0) + 5
                    print(f"\n  Rate limit hit. Waiting {wait_time:.0f}s...")
                    time.sleep(wait_time)
                    continue

                if response.status_code != 200:
                    print(f"  Error: HTTP {response.status_code}")
                    break

                data = response.json()
                items = data.get('items', [])

                if not items:
                    break

                for item in items:
                    # Get raw URL
                    html_url = item.get('html_url', '')
                    if not html_url or html_url in self.seen_urls:
                        continue

                    raw_url = html_url.replace('github.com', 'raw.githubusercontent.com')
                    raw_url = raw_url.replace('/blob/', '/')

                    size = item.get('size', 0)
                    repo = item.get('repository', {}).get('full_name', '')

                    # Filter by size (50 bytes to 50KB)
                    if 50 <= size <= 50000:
                        scripts_found.append({
                            'url': raw_url,
                            'repo': repo,
                            'size': size,
                            'name': item.get('name', '')
                        })
                        self.seen_urls.add(html_url)

                print(f"  Found {len(scripts_found)} scripts (page {page})")
                page += 1
                time.sleep(2)  # Be nice to GitHub

                if len(scripts_found) >= max_results:
                    break

            except Exception as e:
                print(f"  Error: {e}")
                break

        return scripts_found[:max_results]

    def download_script(self, script_info, retries=3):
        """Download a single script."""
        for attempt in range(retries):
            try:
                response = self.session.get(script_info['url'], timeout=10)

                if response.status_code == 200:
                    content = response.text.strip()

                    # Basic validation
                    if len(content) > 50 and ('#!' in content or 'bash' in content.lower()):
                        return content

                elif response.status_code == 404:
                    break  # Don't retry 404s

                time.sleep(1)

            except Exception:
                if attempt == retries - 1:
                    return None
                time.sleep(2)

        return None

    def download_all(self, script_infos):
        """Download all scripts with progress bar."""
        print(f"\nDownloading {len(script_infos)} scripts...")

        scripts = []

        for info in tqdm(script_infos, desc="Downloading"):
            content = self.download_script(info)

            if content:
                scripts.append(content)

            time.sleep(0.5)  # Rate limiting

        return scripts


def get_comprehensive_queries():
    """Get diverse search queries to find variety of bash scripts."""
    return [
        # By topic
        "language:shell topic:bash stars:>10",
        "language:shell topic:devops stars:>5",
        "language:shell topic:automation stars:>5",
        "language:shell topic:sysadmin stars:>5",
        "language:shell topic:docker stars:>5",
        "language:shell topic:kubernetes stars:>5",

        # By use case
        "filename:*.sh backup in:file",
        "filename:*.sh deploy in:file",
        "filename:*.sh install in:file",
        "filename:*.sh setup in:file",
        "filename:*.sh build in:file",
        "filename:*.sh test in:file",
        "filename:*.sh monitor in:file",
        "filename:*.sh script in:file",

        # By content
        "extension:sh curl in:file stars:>5",
        "extension:sh docker in:file stars:>5",
        "extension:sh git in:file stars:>5",
        "extension:sh systemd in:file stars:>5",
        "extension:sh nginx in:file stars:>5",
        "extension:sh mysql in:file stars:>5",

        # Popular repos
        "repo:awesome-lists/awesome-bash extension:sh",
        "repo:dylanaraps/pure-bash-bible extension:sh",
    ]


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Download 10,000+ bash scripts from GitHub')

    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent.parent / 'data_code',
                        help='Output directory')
    parser.add_argument('--target-count', type=int, default=10000,
                        help='Target number of scripts')
    parser.add_argument('--github-token', type=str,
                        default=os.environ.get('GITHUB_TOKEN'),
                        help='GitHub token (or set GITHUB_TOKEN env var)')

    args = parser.parse_args()

    print("="*60)
    print("GitHub Bash Script Downloader")
    print("="*60)
    print(f"Target: {args.target_count:,} scripts")
    print()

    # Create downloader
    downloader = GitHubBashDownloader(token=args.github_token)

    # Get search queries
    queries = get_comprehensive_queries()
    scripts_per_query = (args.target_count // len(queries)) + 100

    print(f"\nUsing {len(queries)} search queries")
    print(f"Target per query: ~{scripts_per_query} scripts")
    print()

    # Search and collect script infos
    all_script_infos = []

    for query in queries:
        script_infos = downloader.search_code(query, max_results=scripts_per_query)
        all_script_infos.extend(script_infos)

        print(f"  Total unique scripts so far: {len(downloader.seen_urls)}")

        if len(downloader.seen_urls) >= args.target_count:
            print(f"\n✓ Reached target of {args.target_count:,} unique scripts!")
            break

        # Rate limiting between queries
        time.sleep(3)

    print(f"\nFound {len(all_script_infos):,} unique scripts to download")

    # Limit to target
    if len(all_script_infos) > args.target_count:
        all_script_infos = all_script_infos[:args.target_count]
        print(f"Limited to {args.target_count:,} scripts")

    # Download all scripts
    scripts = downloader.download_all(all_script_infos)

    print(f"\n✓ Successfully downloaded {len(scripts):,} scripts")

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_file = args.output_dir / "code_data.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'scripts': scripts,
            'count': len(scripts),
            'source': 'github'
        }, f, indent=2)

    # Stats
    total_chars = sum(len(s) for s in scripts)
    total_lines = sum(s.count('\n') + 1 for s in scripts)

    stats = {
        'source': 'github',
        'num_scripts': len(scripts),
        'total_chars': total_chars,
        'total_lines': total_lines,
        'avg_chars': total_chars / len(scripts) if scripts else 0,
        'avg_lines': total_lines / len(scripts) if scripts else 0,
    }

    stats_file = args.output_dir / "stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "="*60)
    print("Download Complete!")
    print("="*60)
    print(f"  Scripts: {stats['num_scripts']:,}")
    print(f"  Lines: {stats['total_lines']:,}")
    print(f"  Characters: {stats['total_chars']:,}")
    print(f"  Avg length: {stats['avg_lines']:.0f} lines")
    print(f"\nSaved to: {output_file}")

    print("\n" + "="*60)
    print("Ready for Training!")
    print("="*60)
    print(f"\nHigh-quality bash script dataset prepared.")
    print(f"Next: Train code model on this data")


if __name__ == '__main__':
    main()
