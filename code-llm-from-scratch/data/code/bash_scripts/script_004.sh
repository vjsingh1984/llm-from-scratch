#!/bin/bash
# Backup rotation script with retention policy

BACKUP_DIR="/backup"
RETENTION_DAYS=7
MONTHLY_KEEP=6
YEARLY_KEEP=2

rotate_backups() {
    cd "$BACKUP_DIR" || exit 1

    # Delete old daily backups
    echo "Removing daily backups older than $RETENTION_DAYS days..."
    find . -name "daily_*.tar.gz" -mtime +$RETENTION_DAYS -delete

    # Keep monthly backups
    echo "Keeping $MONTHLY_KEEP monthly backups..."
    ls -t monthly_*.tar.gz 2>/dev/null | tail -n +$((MONTHLY_KEEP + 1)) | xargs rm -f

    # Keep yearly backups
    echo "Keeping $YEARLY_KEEP yearly backups..."
    ls -t yearly_*.tar.gz 2>/dev/null | tail -n +$((YEARLY_KEEP + 1)) | xargs rm -f
}

create_backup() {
    local type=$1
    local filename="${type}_$(date +%Y%m%d_%H%M%S).tar.gz"

    echo "Creating $type backup: $filename"

    tar -czf "$BACKUP_DIR/$filename" /data 2>/dev/null

    if [ $? -eq 0 ]; then
        echo "Backup created successfully"

        # Generate checksum
        cd "$BACKUP_DIR" && sha256sum "$filename" > "$filename.sha256"
    else
        echo "Backup failed!"
        exit 1
    fi
}

# Determine backup type based on date
DAY=$(date +%d)
MONTH=$(date +%m)

if [ "$DAY" = "01" ] && [ "$MONTH" = "01" ]; then
    create_backup "yearly"
elif [ "$DAY" = "01" ]; then
    create_backup "monthly"
else
    create_backup "daily"
fi

rotate_backups

echo "Backup and rotation complete"
