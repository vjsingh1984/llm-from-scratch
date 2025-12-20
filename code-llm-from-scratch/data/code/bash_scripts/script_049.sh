#!/bin/bash
# Table size analyzer

DB_NAME="${1}"

mysql "$DB_NAME" -e "
    SELECT
        table_name,
        ROUND(((data_length + index_length) / 1024 / 1024), 2) AS 'Size (MB)',
        ROUND((data_length / 1024 / 1024), 2) AS 'Data (MB)',
        ROUND((index_length / 1024 / 1024), 2) AS 'Index (MB)',
        table_rows,
        ROUND((data_length / table_rows), 2) AS 'Avg Row Length'
    FROM information_schema.TABLES
    WHERE table_schema = '$DB_NAME'
    ORDER BY (data_length + index_length) DESC;
"
