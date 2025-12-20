#!/bin/bash
# Analyze slow queries

DB_USER="root"
DB_PASS="$(cat /etc/mysql/root.pass)"
SLOW_QUERY_LOG="/var/log/mysql/slow-query.log"
REPORT_FILE="/tmp/slow-queries-$(date +%Y%m%d).txt"

{
    echo "Slow Query Analysis"
    echo "==================="
    echo "Date: $(date)"
    echo

    if [ ! -f "$SLOW_QUERY_LOG" ]; then
        echo "Error: Slow query log not found"
        exit 1
    fi

    echo "=== Top 10 Slowest Queries ==="
    mysqldumpslow -s t -t 10 "$SLOW_QUERY_LOG"

    echo
    echo "=== Most Frequent Slow Queries ==="
    mysqldumpslow -s c -t 10 "$SLOW_QUERY_LOG"

    echo
    echo "=== Table Statistics ==="
    mysql -u "$DB_USER" -p"$DB_PASS" -e "
        SELECT
            table_schema AS 'Database',
            table_name AS 'Table',
            ROUND(((data_length + index_length) / 1024 / 1024), 2) AS 'Size (MB)',
            table_rows AS 'Rows'
        FROM information_schema.TABLES
        WHERE table_schema NOT IN ('information_schema', 'mysql', 'performance_schema')
        ORDER BY (data_length + index_length) DESC
        LIMIT 10;
    "

    echo
    echo "=== Current Running Queries ==="
    mysql -u "$DB_USER" -p"$DB_PASS" -e "SHOW FULL PROCESSLIST"

} | tee "$REPORT_FILE"

echo
echo "Report saved to: $REPORT_FILE"
