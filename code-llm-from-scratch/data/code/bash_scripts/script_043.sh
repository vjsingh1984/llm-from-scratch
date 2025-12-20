#!/bin/bash
# Database restore script

BACKUP_FILE="${1}"
DB_NAME="${2}"
DB_TYPE="${3:-mysql}"

if [ -z "$BACKUP_FILE" ] || [ -z "$DB_NAME" ]; then
    echo "Usage: $0 <backup-file> <database-name> [mysql|postgresql]"
    exit 1
fi

if [ ! -f "$BACKUP_FILE" ]; then
    echo "Error: Backup file not found: $BACKUP_FILE"
    exit 1
fi

echo "Database Restore"
echo "================"
echo "File: $BACKUP_FILE"
echo "Database: $DB_NAME"
echo "Type: $DB_TYPE"
echo

read -p "This will overwrite database $DB_NAME. Continue? (yes/no) " -r
if [ "$REPLY" != "yes" ]; then
    echo "Restore cancelled"
    exit 0
fi

restore_mysql() {
    echo "Restoring MySQL database..."

    # Create database if it doesn't exist
    mysql -e "CREATE DATABASE IF NOT EXISTS $DB_NAME"

    # Restore
    if [[ "$BACKUP_FILE" == *.gz ]]; then
        gunzip < "$BACKUP_FILE" | mysql "$DB_NAME"
    else
        mysql "$DB_NAME" < "$BACKUP_FILE"
    fi
}

restore_postgresql() {
    echo "Restoring PostgreSQL database..."

    # Drop and recreate database
    dropdb "$DB_NAME" 2>/dev/null
    createdb "$DB_NAME"

    # Restore
    if [[ "$BACKUP_FILE" == *.dump ]]; then
        pg_restore -d "$DB_NAME" "$BACKUP_FILE"
    else
        psql "$DB_NAME" < "$BACKUP_FILE"
    fi
}

case $DB_TYPE in
    mysql)
        restore_mysql
        ;;
    postgresql|postgres)
        restore_postgresql
        ;;
    *)
        echo "Error: Unknown database type: $DB_TYPE"
        exit 1
        ;;
esac

if [ $? -eq 0 ]; then
    echo "✓ Database restored successfully"
else
    echo "✗ Restore failed"
    exit 1
fi
