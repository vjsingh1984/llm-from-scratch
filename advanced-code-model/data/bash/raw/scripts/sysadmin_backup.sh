#!/bin/bash
# System backup script with rotation

set -euo pipefail

BACKUP_DIR="/backup"
SOURCE_DIRS=("/etc" "/home" "/var/www")
RETENTION_DAYS=7
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Starting system backup: $TIMESTAMP"

for SOURCE in "${SOURCE_DIRS[@]}"; do
    BACKUP_NAME=$(basename "$SOURCE")_$TIMESTAMP.tar.gz
    echo "Backing up: $SOURCE"
    tar -czf "$BACKUP_DIR/$BACKUP_NAME" "$SOURCE" 2>/dev/null || true
done

# Cleanup old backups
find "$BACKUP_DIR" -type f -mtime +$RETENTION_DAYS -delete

echo "Backup complete"
