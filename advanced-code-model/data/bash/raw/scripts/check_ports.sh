#!/bin/bash
# Check open ports and services

echo "Open Ports and Services"
echo "======================="
echo ""

# Show listening ports
echo "TCP Ports:"
netstat -tuln | grep LISTEN | awk '{print $4}' | sed 's/.*://' | sort -n | uniq

echo ""
echo "Active Connections:"
netstat -an | grep ESTABLISHED | wc -l

echo ""
echo "Services by Port:"
lsof -i -P -n | grep LISTEN | awk '{print $1, $9}' | sort -u
