#!/bin/bash
# PostgreSQL backup script

PGUSER="postgres"
BACKUP_DIR="/backup/postgresql"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"

# Backup all databases
echo "Backing up all PostgreSQL databases..."

pg_dumpall -U "$PGUSER" | gzip > "$BACKUP_DIR/all-databases_${DATE}.sql.gz"

# Backup individual databases
for db in $(psql -U "$PGUSER" -t -c "SELECT datname FROM pg_database WHERE datname NOT IN ('template0', 'template1', 'postgres')"); do
    db=$(echo "$db" | tr -d ' ')
    echo "Backing up: $db"

    pg_dump -U "$PGUSER" -Fc "$db" > "$BACKUP_DIR/${db}_${DATE}.dump"
done

# Backup globals (users, roles, tablespaces)
pg_dumpall -U "$PGUSER" --globals-only > "$BACKUP_DIR/globals_${DATE}.sql"

# Clean old backups
find "$BACKUP_DIR" -name "*.sql.gz" -mtime +14 -delete
find "$BACKUP_DIR" -name "*.dump" -mtime +14 -delete

echo "✓ PostgreSQL backup complete"
