#!/bin/bash
# Event correlation engine

grep -E "(error|warning)" /var/log/syslog |     awk '{print $1, $2, $5}' |     sort | uniq -c | sort -rn
