#!/bin/bash
# Index optimization

DB_NAME="${1}"

echo "Analyzing indexes for: $DB_NAME"

mysql "$DB_NAME" -e "
    SELECT
        table_schema,
        table_name,
        index_name,
        SEQ_IN_INDEX,
        column_name,
        cardinality
    FROM information_schema.STATISTICS
    WHERE table_schema = '$DB_NAME'
    AND cardinality IS NOT NULL
    ORDER BY table_name, index_name, SEQ_IN_INDEX;
"
