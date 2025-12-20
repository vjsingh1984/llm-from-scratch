#!/bin/bash
# MySQL table optimization

MYSQL_USER="root"
MYSQL_PASS="${MYSQL_PASSWORD}"
DATABASE="$1"

if [ -z "$DATABASE" ]; then
    echo "Usage: $0 <database_name>"
    exit 1
fi

echo "Optimizing tables in database: $DATABASE"

# Get list of tables
TABLES=$(mysql -u"$MYSQL_USER" -p"$MYSQL_PASS" -D"$DATABASE" -e "SHOW TABLES" | tail -n +2)

# Optimize each table
for TABLE in $TABLES; do
    echo "Optimizing table: $TABLE"
    mysql -u"$MYSQL_USER" -p"$MYSQL_PASS" -D"$DATABASE" -e "OPTIMIZE TABLE $TABLE"
done

echo "Optimization complete!"
