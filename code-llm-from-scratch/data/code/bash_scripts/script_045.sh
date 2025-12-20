#!/bin/bash
# Monitor database connections

DB_USER="monitor"
DB_PASS="$(cat /etc/mysql/monitor.pass)"
MAX_CONNECTIONS=150
WARNING_THRESHOLD=120

get_connection_count() {
    mysql -u "$DB_USER" -p"$DB_PASS" -sN -e "SHOW STATUS LIKE 'Threads_connected'" | awk '{print $2}'
}

get_max_used_connections() {
    mysql -u "$DB_USER" -p"$DB_PASS" -sN -e "SHOW STATUS LIKE 'Max_used_connections'" | awk '{print $2}'
}

echo "Database Connection Monitor"
echo "==========================="

CURRENT=$(get_connection_count)
MAX_USED=$(get_max_used_connections)
PERCENT=$((CURRENT * 100 / MAX_CONNECTIONS))

echo "Current connections: $CURRENT"
echo "Max connections: $MAX_CONNECTIONS"
echo "Max used: $MAX_USED"
echo "Usage: ${PERCENT}%"
echo

# Show connection details
echo "Connections by user:"
mysql -u "$DB_USER" -p"$DB_PASS" -e "
    SELECT user, COUNT(*) as connections
    FROM information_schema.PROCESSLIST
    GROUP BY user
    ORDER BY connections DESC;
"

echo
echo "Connections by host:"
mysql -u "$DB_USER" -p"$DB_PASS" -e "
    SELECT host, COUNT(*) as connections
    FROM information_schema.PROCESSLIST
    GROUP BY host
    ORDER BY connections DESC;
"

# Alert if threshold exceeded
if [ "$CURRENT" -gt "$WARNING_THRESHOLD" ]; then
    echo
    echo "⚠ WARNING: Connection count exceeds threshold"
    echo "Current: $CURRENT, Threshold: $WARNING_THRESHOLD"
    exit 1
else
    echo
    echo "✓ Connection count normal"
    exit 0
fi
