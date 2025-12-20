#!/bin/bash
# Advanced backup script with rotation

SOURCE_DIR="${1:-/data}"
BACKUP_ROOT="/backup"
RETENTION_DAYS=7
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$BACKUP_ROOT/$DATE"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup with progress
echo "Starting backup of $SOURCE_DIR..."
tar -czf "$BACKUP_DIR/backup.tar.gz" "$SOURCE_DIR" 2>&1 |     while read line; do
        echo "  $line"
    done

# Calculate checksum
cd "$BACKUP_DIR"
sha256sum backup.tar.gz > backup.tar.gz.sha256

# Remove old backups
echo "Cleaning old backups (older than $RETENTION_DAYS days)..."
find "$BACKUP_ROOT" -maxdepth 1 -type d -mtime +$RETENTION_DAYS -exec rm -rf {} \;

# Report
BACKUP_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
echo "Backup complete!"
echo "  Location: $BACKUP_DIR"
echo "  Size: $BACKUP_SIZE"
echo "  Checksum: $(cat backup.tar.gz.sha256)"
