#!/bin/bash
# PostgreSQL backup with compression

set -euo pipefail

DB_NAME="${1:-mydb}"
BACKUP_DIR="/backup/postgresql"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/${DB_NAME}_${TIMESTAMP}.sql.gz"

mkdir -p "$BACKUP_DIR"

echo "Backing up database: $DB_NAME"

# Create backup with compression
pg_dump -U postgres "$DB_NAME" | gzip > "$BACKUP_FILE"

# Verify backup
if [ -f "$BACKUP_FILE" ] && [ -s "$BACKUP_FILE" ]; then
    echo "Backup successful: $BACKUP_FILE"
    SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
    echo "Size: $SIZE"
else
    echo "Backup failed!"
    exit 1
fi

# Keep only last 7 days
find "$BACKUP_DIR" -name "${DB_NAME}_*.sql.gz" -mtime +7 -delete

echo "Old backups cleaned up"
