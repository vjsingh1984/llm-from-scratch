#!/bin/bash
# Transaction log monitor

echo "Monitoring transaction logs..."

mysql -e "SHOW ENGINE INNODB STATUS\G" | grep -A 20 "TRANSACTIONS"
