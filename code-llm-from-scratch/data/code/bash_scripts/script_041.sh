#!/bin/bash
# MySQL database backup with compression

DB_USER="backup_user"
DB_PASS="$(cat /etc/mysql/backup.pass)"
BACKUP_DIR="/backup/mysql"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"

# Get all databases
DATABASES=$(mysql -u "$DB_USER" -p"$DB_PASS" -e "SHOW DATABASES;" | grep -Ev "(Database|information_schema|performance_schema|mysql)")

for db in $DATABASES; do
    echo "Backing up database: $db"

    mysqldump -u "$DB_USER" -p"$DB_PASS"         --single-transaction         --routines         --triggers         "$db" | gzip > "$BACKUP_DIR/${db}_${DATE}.sql.gz"

    if [ $? -eq 0 ]; then
        echo "✓ $db backed up successfully"
    else
        echo "✗ Failed to backup $db"
    fi
done

# Remove backups older than 7 days
find "$BACKUP_DIR" -name "*.sql.gz" -mtime +7 -delete

echo "Database backup complete"
