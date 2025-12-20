#!/bin/bash
# Database replication checker

MASTER_HOST="db-master"
SLAVE_HOST="db-slave"

echo "Checking replication status..."

# Check slave status
mysql -h "$SLAVE_HOST" -e "SHOW SLAVE STATUS\G" | grep -E "(Slave_IO_Running|Slave_SQL_Running|Seconds_Behind_Master)"

echo "✓ Replication check complete"
