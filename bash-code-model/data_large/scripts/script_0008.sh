#!/bin/bash
# Server monitoring with alerts

THRESHOLD_CPU=80
THRESHOLD_MEM=90
THRESHOLD_DISK=85
ALERT_EMAIL="admin@example.com"

send_alert() {
    local subject="$1"
    local message="$2"
    echo "$message" | mail -s "$subject" "$ALERT_EMAIL"
}

# Check CPU
cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1 | cut -d'.' -f1)
if [ $cpu_usage -gt $THRESHOLD_CPU ]; then
    send_alert "High CPU Usage Alert" "CPU usage is ${cpu_usage}%"
fi

# Check Memory
mem_usage=$(free | grep Mem | awk '{print ($3/$2) * 100.0}' | cut -d'.' -f1)
if [ $mem_usage -gt $THRESHOLD_MEM ]; then
    send_alert "High Memory Usage Alert" "Memory usage is ${mem_usage}%"
fi

# Check Disk
disk_usage=$(df -h / | tail -1 | awk '{print $5}' | cut -d'%' -f1)
if [ $disk_usage -gt $THRESHOLD_DISK ]; then
    send_alert "High Disk Usage Alert" "Disk usage is ${disk_usage}%"
fi

echo "Monitoring complete at $(date)"
