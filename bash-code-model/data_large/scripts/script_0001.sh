#!/bin/bash
# System health check script

echo "=== System Health Report ==="
echo "Date: $(date)"
echo

echo "--- CPU Usage ---"
top -bn1 | grep "Cpu(s)" | awk '{print "CPU Usage: " $2 + $4 "%"}'

echo
echo "--- Memory Usage ---"
free -h | grep Mem | awk '{print "Total: " $2 ", Used: " $3 ", Free: " $4}'

echo
echo "--- Disk Usage ---"
df -h | grep -vE '^Filesystem|tmpfs|cdrom'

echo
echo "--- Top 5 Processes by CPU ---"
ps aux --sort=-%cpu | head -6

echo
echo "--- Network Connections ---"
netstat -an | grep ESTABLISHED | wc -l | awk '{print "Active connections: " $1}'
