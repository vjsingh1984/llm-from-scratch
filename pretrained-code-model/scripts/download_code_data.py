"""
Download high-quality bash script data for code fine-tuning.

Sources:
1. Use existing curated examples from bash-code-model
2. Download from GitHub (requires GITHUB_TOKEN for higher rate limits)
3. Public examples with production-quality scripts
"""

import sys
from pathlib import Path
import json
import shutil

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def use_existing_bash_data(output_dir: Path):
    """Use existing bash script data from bash-code-model project."""
    # Check for large dataset
    large_data = Path(__file__).parent.parent.parent / 'bash-code-model' / 'data_large' / 'bash_scripts.json'

    # Check for small dataset
    small_data = Path(__file__).parent.parent.parent / 'bash-code-model' / 'data' / 'bash_scripts.json'

    source_file = None

    if large_data.exists():
        source_file = large_data
        print(f"Found large bash dataset: {large_data}")
    elif small_data.exists():
        source_file = small_data
        print(f"Found small bash dataset: {small_data}")
    else:
        print("No existing bash data found in bash-code-model project")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy to our project
    dest_file = output_dir / "code_data.json"

    with open(source_file) as f:
        data = json.load(f)

    scripts = data['scripts']
    print(f"  Scripts found: {len(scripts)}")

    # Save to our format
    with open(dest_file, 'w') as f:
        json.dump({
            'scripts': scripts,
            'count': len(scripts),
            'source': 'bash-code-model'
        }, f, indent=2)

    # Stats
    total_chars = sum(len(s) for s in scripts)
    total_lines = sum(s.count('\n') + 1 for s in scripts)

    stats = {
        'source': 'bash-code-model',
        'num_scripts': len(scripts),
        'total_chars': total_chars,
        'total_lines': total_lines,
        'avg_chars': total_chars / len(scripts) if scripts else 0,
        'avg_lines': total_lines / len(scripts) if scripts else 0,
    }

    stats_file = output_dir / "stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "="*60)
    print("Bash Scripts Prepared!")
    print("="*60)
    print(f"  Scripts: {stats['num_scripts']}")
    print(f"  Lines: {stats['total_lines']:,}")
    print(f"  Characters: {stats['total_chars']:,}")
    print(f"  Avg length: {stats['avg_lines']:.0f} lines")
    print(f"\nSaved to: {dest_file}")

    return True


def create_comprehensive_bash_dataset(output_dir: Path):
    """
    Create a comprehensive high-quality bash script dataset.

    These are production-ready examples covering:
    - System administration
    - DevOps automation
    - Data processing
    - Network operations
    - Security tasks
    """
    print("="*60)
    print("Creating Comprehensive Bash Dataset")
    print("="*60)
    print()

    scripts = []

    # Load from bash-code-model if exists
    large_data = Path(__file__).parent.parent.parent / 'bash-code-model' / 'data_large' / 'bash_scripts.json'

    if large_data.exists():
        with open(large_data) as f:
            data = json.load(f)
            scripts.extend(data['scripts'])
        print(f"Loaded {len(scripts)} scripts from bash-code-model")

    # Add more high-quality examples
    additional_scripts = [
        # Advanced backup with error handling
        """#!/bin/bash
# Production backup script with error handling and notification
set -euo pipefail

BACKUP_SOURCE="${1:-/data}"
BACKUP_DEST="${2:-/backup}"
DATE=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/var/log/backup_${DATE}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log "ERROR: Backup failed with exit code $exit_code"
        # Send notification
        echo "Backup failed" | mail -s "Backup Alert" admin@example.com
    fi
    exit $exit_code
}

trap cleanup EXIT

log "Starting backup of $BACKUP_SOURCE"

# Create backup directory
mkdir -p "$BACKUP_DEST"

# Perform backup
tar -czf "$BACKUP_DEST/backup_${DATE}.tar.gz" "$BACKUP_SOURCE" 2>&1 | tee -a "$LOG_FILE"

# Verify backup
if tar -tzf "$BACKUP_DEST/backup_${DATE}.tar.gz" > /dev/null 2>&1; then
    log "Backup verified successfully"
else
    log "ERROR: Backup verification failed"
    exit 1
fi

# Cleanup old backups (keep last 7 days)
find "$BACKUP_DEST" -name "backup_*.tar.gz" -mtime +7 -delete

log "Backup completed successfully"
""",

        # System monitoring with metrics collection
        """#!/bin/bash
# Collect system metrics and generate report

REPORT_DIR="/var/reports"
DATE=$(date +%Y%m%d)
REPORT_FILE="$REPORT_DIR/system_metrics_$DATE.json"

mkdir -p "$REPORT_DIR"

# Collect metrics
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
MEM_TOTAL=$(free -m | awk 'NR==2{print $2}')
MEM_USED=$(free -m | awk 'NR==2{print $3}')
MEM_PCT=$(awk "BEGIN {printf \"%.2f\", ($MEM_USED/$MEM_TOTAL)*100}")

DISK_USAGE=$(df -h / | awk 'NR==2{print $5}' | sed 's/%//')
LOAD_AVG=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')

# Network stats
RX_BYTES=$(cat /sys/class/net/eth0/statistics/rx_bytes 2>/dev/null || echo 0)
TX_BYTES=$(cat /sys/class/net/eth0/statistics/tx_bytes 2>/dev/null || echo 0)

# Generate JSON report
cat > "$REPORT_FILE" <<EOF
{
  "timestamp": "$(date -Iseconds)",
  "hostname": "$(hostname)",
  "metrics": {
    "cpu_usage_percent": $CPU_USAGE,
    "memory": {
      "total_mb": $MEM_TOTAL,
      "used_mb": $MEM_USED,
      "usage_percent": $MEM_PCT
    },
    "disk_usage_percent": $DISK_USAGE,
    "load_average": $LOAD_AVG,
    "network": {
      "rx_bytes": $RX_BYTES,
      "tx_bytes": $TX_BYTES
    }
  }
}
EOF

echo "Metrics saved to $REPORT_FILE"

# Alert if thresholds exceeded
if (( $(echo "$CPU_USAGE > 80" | bc -l) )); then
    echo "ALERT: High CPU usage: ${CPU_USAGE}%"
fi

if (( $(echo "$MEM_PCT > 90" | bc -l) )); then
    echo "ALERT: High memory usage: ${MEM_PCT}%"
fi
""",

        # Log rotation and analysis
        """#!/bin/bash
# Intelligent log rotation with analysis

LOG_DIR="/var/log/app"
ARCHIVE_DIR="/var/log/archive"
MAX_SIZE_MB=100
RETENTION_DAYS=30

rotate_logs() {
    local log_file="$1"
    local size_mb=$(du -m "$log_file" | cut -f1)

    if [ $size_mb -gt $MAX_SIZE_MB ]; then
        echo "Rotating $log_file (${size_mb}MB)"

        # Create archive directory
        mkdir -p "$ARCHIVE_DIR"

        # Compress and move
        gzip -c "$log_file" > "$ARCHIVE_DIR/$(basename $log_file)_$(date +%Y%m%d_%H%M%S).gz"

        # Truncate original
        > "$log_file"

        echo "Rotated to archive"
    fi
}

# Analyze logs before rotation
analyze_logs() {
    local log_file="$1"

    echo "Log Analysis: $(basename $log_file)"
    echo "  Total lines: $(wc -l < "$log_file")"
    echo "  Errors: $(grep -c ERROR "$log_file" || echo 0)"
    echo "  Warnings: $(grep -c WARN "$log_file" || echo 0)"
    echo

    # Extract top errors
    echo "  Top errors:"
    grep ERROR "$log_file" | sort | uniq -c | sort -rn | head -3 || echo "    None"
}

# Process all log files
for log_file in "$LOG_DIR"/*.log; do
    if [ -f "$log_file" ]; then
        analyze_logs "$log_file"
        rotate_logs "$log_file"
    fi
done

# Cleanup old archives
find "$ARCHIVE_DIR" -name "*.gz" -mtime +$RETENTION_DAYS -delete

echo "Log rotation complete"
""",

        # Deployment script with health checks
        """#!/bin/bash
# Zero-downtime deployment with health checks

APP_NAME="myapp"
DEPLOY_DIR="/opt/$APP_NAME"
NEW_VERSION="$1"
HEALTH_URL="http://localhost:8080/health"
MAX_RETRIES=30

if [ -z "$NEW_VERSION" ]; then
    echo "Usage: $0 <version>"
    exit 1
fi

health_check() {
    local retries=0

    while [ $retries -lt $MAX_RETRIES ]; do
        if curl -sf "$HEALTH_URL" > /dev/null; then
            echo "Health check passed"
            return 0
        fi

        echo "Health check attempt $((retries + 1))/$MAX_RETRIES..."
        sleep 2
        retries=$((retries + 1))
    done

    echo "Health check failed after $MAX_RETRIES attempts"
    return 1
}

rollback() {
    echo "Rolling back to previous version..."
    systemctl stop "$APP_NAME"
    mv "$DEPLOY_DIR/current" "$DEPLOY_DIR/failed_$NEW_VERSION"
    mv "$DEPLOY_DIR/previous" "$DEPLOY_DIR/current"
    systemctl start "$APP_NAME"
    echo "Rollback complete"
}

# Backup current version
if [ -d "$DEPLOY_DIR/current" ]; then
    mv "$DEPLOY_DIR/current" "$DEPLOY_DIR/previous"
fi

# Deploy new version
echo "Deploying version $NEW_VERSION..."
git clone --branch "$NEW_VERSION" https://github.com/user/app.git "$DEPLOY_DIR/current"

cd "$DEPLOY_DIR/current"

# Build
echo "Building..."
make build || { rollback; exit 1; }

# Restart service
echo "Restarting service..."
systemctl restart "$APP_NAME"

# Health check
echo "Performing health check..."
if health_check; then
    echo "Deployment successful!"
    rm -rf "$DEPLOY_DIR/previous"
else
    rollback
    exit 1
fi
""",

        # Database maintenance script
        """#!/bin/bash
# Automated database maintenance

DB_NAME="${1:-production}"
DB_USER="${2:-admin}"
BACKUP_DIR="/var/backups/db"
DATE=$(date +%Y%m%d)

mkdir -p "$BACKUP_DIR"

# Backup
echo "Creating backup..."
pg_dump -U "$DB_USER" "$DB_NAME" | gzip > "$BACKUP_DIR/${DB_NAME}_${DATE}.sql.gz"

# Verify backup
if gunzip -t "$BACKUP_DIR/${DB_NAME}_${DATE}.sql.gz"; then
    echo "Backup verified: $BACKUP_DIR/${DB_NAME}_${DATE}.sql.gz"
else
    echo "Backup verification failed!"
    exit 1
fi

# Maintenance
echo "Running VACUUM ANALYZE..."
psql -U "$DB_USER" -d "$DB_NAME" -c "VACUUM ANALYZE;"

# Reindex
echo "Reindexing..."
psql -U "$DB_USER" -d "$DB_NAME" -c "REINDEX DATABASE $DB_NAME;"

# Check for bloat
echo "Checking for table bloat..."
psql -U "$DB_USER" -d "$DB_NAME" -c "
SELECT
  schemaname || '.' || tablename AS table,
  pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
LIMIT 10;
"

# Cleanup old backups
find "$BACKUP_DIR" -name "${DB_NAME}_*.sql.gz" -mtime +7 -delete

echo "Database maintenance complete"
"""
    ]

    scripts.extend(additional_scripts)
    print(f"Added {len(additional_scripts)} additional high-quality scripts")
    print(f"Total scripts: {len(scripts)}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    dest_file = output_dir / "code_data.json"
    with open(dest_file, 'w') as f:
        json.dump({
            'scripts': scripts,
            'count': len(scripts),
            'source': 'comprehensive'
        }, f, indent=2)

    # Stats
    total_chars = sum(len(s) for s in scripts)
    total_lines = sum(s.count('\n') + 1 for s in scripts)

    stats = {
        'source': 'comprehensive',
        'num_scripts': len(scripts),
        'total_chars': total_chars,
        'total_lines': total_lines,
        'avg_chars': total_chars / len(scripts) if scripts else 0,
        'avg_lines': total_lines / len(scripts) if scripts else 0,
    }

    stats_file = output_dir / "stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "="*60)
    print("Comprehensive Bash Dataset Created!")
    print("="*60)
    print(f"  Scripts: {stats['num_scripts']}")
    print(f"  Lines: {stats['total_lines']:,}")
    print(f"  Characters: {stats['total_chars']:,}")
    print(f"  Avg length: {stats['avg_lines']:.0f} lines")
    print(f"\nSaved to: {dest_file}")

    return True


def check_existing(data_dir: Path):
    """Check if code data already exists."""
    json_file = data_dir / "code_data.json"
    stats_file = data_dir / "stats.json"

    if json_file.exists() and stats_file.exists():
        with open(stats_file) as f:
            stats = json.load(f)

        print("="*60)
        print("Existing Code Data Found!")
        print("="*60)
        print(f"  Source: {stats['source']}")
        print(f"  Scripts: {stats['num_scripts']}")
        print(f"  Lines: {stats['total_lines']:,}")
        print(f"  Location: {json_file}")
        print()

        return True

    return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Download bash script data')

    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent.parent / 'data_code',
                        help='Output directory')
    parser.add_argument('--source', type=str,
                        choices=['existing', 'comprehensive', 'auto'],
                        default='auto',
                        help='Data source')
    parser.add_argument('--force', action='store_true',
                        help='Force download even if data exists')

    args = parser.parse_args()

    print("\n")
    print("="*60)
    print("Code Data Download")
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

    if args.source == 'existing':
        success = use_existing_bash_data(args.output_dir)

    elif args.source == 'comprehensive':
        success = create_comprehensive_bash_dataset(args.output_dir)

    else:  # auto
        # Try comprehensive (includes existing + more)
        print("Creating comprehensive high-quality bash dataset...")
        print()
        success = create_comprehensive_bash_dataset(args.output_dir)

    if success:
        print("\n" + "="*60)
        print("Ready for Stage 2: Code Fine-tuning")
        print("="*60)
        print("\nNext step:")
        print("  python scripts/train_code.py")
    else:
        print("\nFailed to prepare code data.")


if __name__ == '__main__':
    main()
