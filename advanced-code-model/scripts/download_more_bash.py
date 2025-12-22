#!/usr/bin/env python3
"""
Download More Bash Training Data

This script downloads additional bash script repositories to improve
Stage 2 training quality. The current ~9MB dataset is too small for
a 1B parameter model.

Target: 50-100MB of high-quality bash scripts
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import List, Tuple
import random

# Additional high-quality bash repositories
ADDITIONAL_REPOS = [
    # DevOps & Cloud
    ("https://github.com/kubernetes/kubernetes", "kubernetes", "*.sh"),
    ("https://github.com/docker/docker-ce", "docker-ce", "*.sh"),
    ("https://github.com/hashicorp/terraform", "terraform", "*.sh"),
    ("https://github.com/ansible/ansible", "ansible", "*.sh"),
    ("https://github.com/jenkinsci/jenkins", "jenkins", "*.sh"),

    # System Administration
    ("https://github.com/ohmyzsh/ohmyzsh", "ohmyzsh", "*.sh"),
    ("https://github.com/robbyrussell/oh-my-zsh", "oh-my-zsh", "*.zsh"),
    ("https://github.com/junegunn/fzf", "fzf", "*.sh"),
    ("https://github.com/tmux-plugins/tpm", "tpm", "*.sh"),
    ("https://github.com/romkatv/powerlevel10k", "powerlevel10k", "*.zsh"),

    # Build Systems
    ("https://github.com/gradle/gradle", "gradle", "*.sh"),
    ("https://github.com/apache/maven", "maven", "*.sh"),
    ("https://github.com/npm/cli", "npm-cli", "*.sh"),

    # Data & ML Ops
    ("https://github.com/apache/spark", "spark", "*.sh"),
    ("https://github.com/apache/hadoop", "hadoop", "*.sh"),
    ("https://github.com/apache/kafka", "kafka", "*.sh"),
    ("https://github.com/apache/airflow", "airflow", "*.sh"),

    # Networking & Security
    ("https://github.com/nginx/nginx", "nginx", "*.sh"),
    ("https://github.com/traefik/traefik", "traefik", "*.sh"),
    ("https://github.com/certbot/certbot", "certbot", "*.sh"),

    # Databases
    ("https://github.com/postgres/postgres", "postgres", "*.sh"),
    ("https://github.com/mysql/mysql-server", "mysql", "*.sh"),
    ("https://github.com/redis/redis", "redis", "*.sh"),
    ("https://github.com/mongodb/mongo", "mongo", "*.sh"),

    # Utilities
    ("https://github.com/dylanaraps/neofetch", "neofetch", "*.sh"),
    ("https://github.com/sharkdp/bat", "bat", "*.sh"),
    ("https://github.com/BurntSushi/ripgrep", "ripgrep", "*.sh"),
    ("https://github.com/stedolan/jq", "jq", "*.sh"),

    # CI/CD
    ("https://github.com/github/super-linter", "super-linter", "*.sh"),
    ("https://github.com/reviewdog/reviewdog", "reviewdog", "*.sh"),

    # Cloud Provider CLIs
    ("https://github.com/aws/aws-cli", "aws-cli", "*.sh"),
    ("https://github.com/Azure/azure-cli", "azure-cli", "*.sh"),
    ("https://github.com/GoogleCloudPlatform/cloud-sdk-docker", "gcloud", "*.sh"),
]

# Curated bash examples for common patterns
BASH_EXAMPLES = '''#!/bin/bash
# Example 1: Safe script template
set -euo pipefail
IFS=$'\\n\\t'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Logging
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

error() {
    echo "[ERROR] $*" >&2
    exit 1
}

# Cleanup on exit
cleanup() {
    log "Cleaning up..."
    rm -rf "$TEMP_DIR" 2>/dev/null || true
}
trap cleanup EXIT

# Main
main() {
    log "Starting script..."
    # Your code here
}

main "$@"

#!/bin/bash
# Example 2: Argument parsing
usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Options:
    -h, --help      Show this help message
    -v, --verbose   Enable verbose output
    -f, --file      Input file path
    -o, --output    Output directory
EOF
    exit 1
}

VERBOSE=false
INPUT_FILE=""
OUTPUT_DIR="."

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -f|--file)
            INPUT_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

[[ -z "$INPUT_FILE" ]] && error "Input file required"
[[ -f "$INPUT_FILE" ]] || error "File not found: $INPUT_FILE"

#!/bin/bash
# Example 3: Docker operations
docker_build() {
    local image_name="$1"
    local dockerfile="${2:-Dockerfile}"

    docker build -t "$image_name" -f "$dockerfile" .
}

docker_push() {
    local image_name="$1"
    local registry="${2:-docker.io}"

    docker tag "$image_name" "$registry/$image_name"
    docker push "$registry/$image_name"
}

docker_cleanup() {
    # Remove dangling images
    docker image prune -f

    # Remove stopped containers
    docker container prune -f

    # Remove unused volumes
    docker volume prune -f
}

#!/bin/bash
# Example 4: Kubernetes operations
kubectl_apply() {
    local manifest="$1"
    local namespace="${2:-default}"

    kubectl apply -f "$manifest" -n "$namespace"
}

kubectl_wait_ready() {
    local resource="$1"
    local namespace="${2:-default}"
    local timeout="${3:-300s}"

    kubectl wait --for=condition=ready "$resource" \\
        -n "$namespace" \\
        --timeout="$timeout"
}

kubectl_rollout_status() {
    local deployment="$1"
    local namespace="${2:-default}"

    kubectl rollout status deployment/"$deployment" -n "$namespace"
}

#!/bin/bash
# Example 5: AWS operations
aws_s3_sync() {
    local source="$1"
    local dest="$2"
    local profile="${3:-default}"

    aws s3 sync "$source" "$dest" \\
        --profile "$profile" \\
        --delete \\
        --exclude ".git/*"
}

aws_ec2_list() {
    local profile="${1:-default}"
    local region="${2:-us-east-1}"

    aws ec2 describe-instances \\
        --profile "$profile" \\
        --region "$region" \\
        --query 'Reservations[].Instances[].[InstanceId,State.Name,Tags[?Key==`Name`].Value|[0]]' \\
        --output table
}

aws_lambda_invoke() {
    local function_name="$1"
    local payload="$2"

    aws lambda invoke \\
        --function-name "$function_name" \\
        --payload "$payload" \\
        --cli-binary-format raw-in-base64-out \\
        /dev/stdout
}

#!/bin/bash
# Example 6: Git operations
git_setup_hooks() {
    local hooks_dir=".git/hooks"

    # Pre-commit hook
    cat > "$hooks_dir/pre-commit" << 'HOOK'
#!/bin/bash
# Run linting before commit
npm run lint || exit 1
npm run test || exit 1
HOOK
    chmod +x "$hooks_dir/pre-commit"
}

git_create_release() {
    local version="$1"
    local message="${2:-Release $version}"

    git tag -a "v$version" -m "$message"
    git push origin "v$version"
}

git_squash_commits() {
    local num_commits="$1"
    git rebase -i "HEAD~$num_commits"
}

#!/bin/bash
# Example 7: Database backup
backup_postgres() {
    local db_name="$1"
    local backup_dir="${2:-/backups}"
    local date_stamp=$(date +%Y%m%d_%H%M%S)

    pg_dump "$db_name" | gzip > "$backup_dir/${db_name}_${date_stamp}.sql.gz"
}

backup_mysql() {
    local db_name="$1"
    local backup_dir="${2:-/backups}"
    local date_stamp=$(date +%Y%m%d_%H%M%S)

    mysqldump "$db_name" | gzip > "$backup_dir/${db_name}_${date_stamp}.sql.gz"
}

restore_postgres() {
    local backup_file="$1"
    local db_name="$2"

    gunzip -c "$backup_file" | psql "$db_name"
}

#!/bin/bash
# Example 8: System monitoring
check_disk_space() {
    local threshold="${1:-80}"

    df -h | awk -v threshold="$threshold" '
        NR>1 && $5+0 > threshold {
            print "WARNING: " $6 " is " $5 " full"
        }
    '
}

check_memory() {
    free -m | awk '
        /^Mem:/ {
            used_pct = ($3/$2) * 100
            printf "Memory: %.1f%% used (%dMB / %dMB)\\n", used_pct, $3, $2
        }
    '
}

check_cpu_load() {
    local threshold="${1:-80}"
    local load=$(uptime | awk -F'load average:' '{print $2}' | cut -d, -f1 | tr -d ' ')

    echo "Current load: $load"
}

#!/bin/bash
# Example 9: File operations
find_large_files() {
    local dir="${1:-.}"
    local size="${2:-100M}"

    find "$dir" -type f -size "+$size" -exec ls -lh {} \\; | sort -k5 -h
}

find_recent_files() {
    local dir="${1:-.}"
    local days="${2:-7}"

    find "$dir" -type f -mtime "-$days" -ls
}

safe_remove() {
    local target="$1"

    if [[ -e "$target" ]]; then
        rm -rf "$target"
        echo "Removed: $target"
    else
        echo "Not found: $target"
    fi
}

#!/bin/bash
# Example 10: Network operations
check_port() {
    local host="$1"
    local port="$2"

    if nc -z "$host" "$port" 2>/dev/null; then
        echo "Port $port on $host is OPEN"
        return 0
    else
        echo "Port $port on $host is CLOSED"
        return 1
    fi
}

wait_for_service() {
    local host="$1"
    local port="$2"
    local timeout="${3:-60}"
    local elapsed=0

    echo "Waiting for $host:$port..."
    while ! nc -z "$host" "$port" 2>/dev/null; do
        sleep 1
        elapsed=$((elapsed + 1))
        if [[ $elapsed -ge $timeout ]]; then
            echo "Timeout waiting for $host:$port"
            return 1
        fi
    done
    echo "Service $host:$port is ready"
}

curl_with_retry() {
    local url="$1"
    local max_retries="${2:-3}"
    local retry=0

    while [[ $retry -lt $max_retries ]]; do
        if curl -sf "$url" > /dev/null; then
            return 0
        fi
        retry=$((retry + 1))
        echo "Retry $retry/$max_retries..."
        sleep $((retry * 2))
    done
    return 1
}
'''


def clone_repo_sparse(url: str, name: str, pattern: str, output_dir: Path) -> int:
    """Clone only shell scripts from a repository."""
    repo_dir = output_dir / name

    if repo_dir.exists():
        print(f"  Skipping {name} (already exists)")
        return 0

    try:
        # Shallow clone with sparse checkout
        subprocess.run([
            "git", "clone", "--depth", "1", "--filter=blob:none",
            "--sparse", url, str(repo_dir)
        ], check=True, capture_output=True, timeout=120)

        # Set sparse checkout to only include shell scripts
        subprocess.run([
            "git", "-C", str(repo_dir), "sparse-checkout", "set",
            "--cone", "scripts", "bin", "hack", "tools", "contrib"
        ], check=True, capture_output=True, timeout=30)

        # Count shell files
        count = len(list(repo_dir.rglob("*.sh"))) + len(list(repo_dir.rglob("*.bash")))
        print(f"  ✓ {name}: {count} shell scripts")
        return count

    except subprocess.TimeoutExpired:
        print(f"  ✗ {name}: Timeout")
        shutil.rmtree(repo_dir, ignore_errors=True)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"  ✗ {name}: Clone failed")
        shutil.rmtree(repo_dir, ignore_errors=True)
        return 0


def collect_bash_scripts(repos_dir: Path, output_file: Path) -> Tuple[int, int]:
    """Collect all bash scripts into a single file."""
    all_scripts = []
    total_files = 0

    # Collect from repos
    for ext in ["*.sh", "*.bash", "*.zsh"]:
        for script_path in repos_dir.rglob(ext):
            try:
                content = script_path.read_text(encoding="utf-8", errors="ignore")
                # Filter: must start with shebang or be non-trivial
                if content.strip() and (
                    content.startswith("#!") or
                    len(content) > 100
                ):
                    all_scripts.append(f"# File: {script_path.name}\n{content}\n")
                    total_files += 1
            except Exception:
                continue

    # Add curated examples
    all_scripts.append(BASH_EXAMPLES)

    # Shuffle and write
    random.shuffle(all_scripts)
    output_file.write_text("\n\n".join(all_scripts), encoding="utf-8")

    total_size = output_file.stat().st_size / (1024 * 1024)
    return total_files, total_size


def main():
    project_root = Path(__file__).parent.parent
    repos_dir = project_root / "data" / "bash" / "raw" / "additional_repos"
    output_dir = project_root / "data" / "raw"

    repos_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Downloading Additional Bash Training Data")
    print("=" * 60)
    print()

    # Download repos
    print("Cloning repositories...")
    total_scripts = 0

    for url, name, pattern in ADDITIONAL_REPOS[:15]:  # Limit to first 15 for speed
        count = clone_repo_sparse(url, name, pattern, repos_dir)
        total_scripts += count

    print(f"\n✓ Downloaded {total_scripts} scripts from repositories")

    # Collect into single file
    print("\nCollecting scripts...")
    output_file = output_dir / "additional_bash_scripts.txt"
    num_files, size_mb = collect_bash_scripts(repos_dir, output_file)

    print(f"✓ Collected {num_files} scripts ({size_mb:.1f} MB)")
    print(f"✓ Saved to: {output_file}")

    # Instructions
    print("\n" + "=" * 60)
    print("Next Steps")
    print("=" * 60)
    print("""
1. Prepare the additional data:
   python3 scripts/prepare_additional_code_data.py

2. Resume Stage 2 training:
   python3 scripts/train.py \\
     --stage code \\
     --checkpoint models/code_model_best.pth \\
     --batch-size 2 --num-epochs 10 \\
     --learning-rate 3e-6 \\
     --use-rmsnorm --use-rope --use-compile --use-amp
""")


if __name__ == "__main__":
    main()
