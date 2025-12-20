#!/bin/bash
# Metric dashboard generator

echo "System Metrics Dashboard"
echo "========================"
echo "Date: $(date)"
echo

echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}')"
echo "Memory: $(free -h | awk 'NR==2{print $3 "/" $2}')"
echo "Disk: $(df -h / | awk 'NR==2{print $5}')"
echo "Load: $(uptime | awk -F'load average:' '{print $2}')"
