#!/bin/bash
# System monitoring dashboard

while true; do
    clear
    echo "==================================="
    echo "   SYSTEM MONITOR - $(date)"
    echo "==================================="
    echo ""
    
    # CPU Usage
    echo "CPU Usage:"
    top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*//" | awk '{print 100 - $1"%"}'
    echo ""
    
    # Memory
    echo "Memory Usage:"
    free -h | awk 'NR==2{printf "Used: %s/%s (%.2f%%)
", $3,$2,$3*100/$2 }'
    echo ""
    
    # Disk
    echo "Disk Usage:"
    df -h / | awk 'NR==2{printf "Used: %s/%s (%s)
", $3,$2,$5}'
    echo ""
    
    # Load Average
    echo "Load Average:"
    uptime | awk -F'load average:' '{print $2}'
    echo ""
    
    # Top Processes
    echo "Top 5 CPU Processes:"
    ps aux --sort=-%cpu | head -6 | tail -5
    echo ""
    
    sleep 5
done
