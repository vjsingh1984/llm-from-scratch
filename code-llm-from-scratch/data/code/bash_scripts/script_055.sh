#!/bin/bash
# Schema diff checker

DB1="${1}"
DB2="${2}"

echo "Comparing schemas: $DB1 vs $DB2"

diff <(mysqldump --no-data "$DB1") <(mysqldump --no-data "$DB2")
