#!/bin/bash
# PostgreSQL database maintenance

DB_NAME="${1:-mydb}"
DB_USER="${2:-postgres}"

echo "=== Database Maintenance: $DB_NAME ==="

# Backup
echo "Creating backup..."
BACKUP_FILE="backup_${DB_NAME}_$(date +%Y%m%d).sql"
pg_dump -U "$DB_USER" "$DB_NAME" > "$BACKUP_FILE"
gzip "$BACKUP_FILE"
echo "  Backup: ${BACKUP_FILE}.gz"

# Vacuum
echo "Running VACUUM..."
psql -U "$DB_USER" -d "$DB_NAME" -c "VACUUM ANALYZE;"

# Reindex
echo "Reindexing..."
psql -U "$DB_USER" -d "$DB_NAME" -c "REINDEX DATABASE $DB_NAME;"

# Statistics
echo
echo "=== Database Statistics ==="
psql -U "$DB_USER" -d "$DB_NAME" -c "    SELECT schemaname, tablename,
           pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
    FROM pg_tables
    WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
    LIMIT 10;"

echo "Maintenance complete"
