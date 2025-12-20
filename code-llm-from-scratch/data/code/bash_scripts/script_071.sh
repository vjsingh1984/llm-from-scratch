#!/bin/bash
# System resource monitoring with alerts

CPU_THRESHOLD=80
MEM_THRESHOLD=90
DISK_THRESHOLD=85

check_cpu() {
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1 | cut -d'.' -f1)

    if [ "$CPU_USAGE" -gt "$CPU_THRESHOLD" ]; then
        echo "ALERT: High CPU usage: ${CPU_USAGE}%"
        top -bn1 | head -20
        return 1
    fi

    echo "CPU: ${CPU_USAGE}% (OK)"
    return 0
}

check_memory() {
    MEM_USAGE=$(free | grep Mem | awk '{printf("%.0f", ($3/$2) * 100)}')

    if [ "$MEM_USAGE" -gt "$MEM_THRESHOLD" ]; then
        echo "ALERT: High memory usage: ${MEM_USAGE}%"
        ps aux --sort=-%mem | head -10
        return 1
    fi

    echo "Memory: ${MEM_USAGE}% (OK)"
    return 0
}

check_disk() {
    DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}' | cut -d'%' -f1)

    if [ "$DISK_USAGE" -gt "$DISK_THRESHOLD" ]; then
        echo "ALERT: High disk usage: ${DISK_USAGE}%"
        du -sh /* 2>/dev/null | sort -hr | head -10
        return 1
    fi

    echo "Disk: ${DISK_USAGE}% (OK)"
    return 0
}

echo "=== System Resource Check ==="
echo "Thresholds: CPU:$CPU_THRESHOLD% MEM:$MEM_THRESHOLD% DISK:$DISK_THRESHOLD%"
echo

ALERTS=0

check_cpu || ((ALERTS++))
check_memory || ((ALERTS++))
check_disk || ((ALERTS++))

echo
if [ $ALERTS -gt 0 ]; then
    echo "Status: $ALERTS alert(s) detected"
    exit 1
else
    echo "Status: All systems normal"
    exit 0
fi
