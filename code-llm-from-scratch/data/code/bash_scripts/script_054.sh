#!/bin/bash
# Database deadlock detector

mysql -e "SHOW ENGINE INNODB STATUS\G" | grep -A 50 "LATEST DETECTED DEADLOCK"
