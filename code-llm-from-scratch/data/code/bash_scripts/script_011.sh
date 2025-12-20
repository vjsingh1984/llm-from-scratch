#!/bin/bash
# Create system performance baseline

OUTPUT_FILE="/var/log/performance-baseline-$(date +%Y%m%d).log"

{
    echo "System Performance Baseline"
    echo "==========================="
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
    echo

    echo "=== CPU Information ==="
    lscpu | grep -E "Model name|CPU\(s\)|MHz|Cache"
    echo

    echo "=== Memory Information ==="
    free -h
    echo

    echo "=== Disk Information ==="
    df -h
    echo

    echo "=== Disk I/O ==="
    iostat -x 1 5
    echo

    echo "=== Network Interfaces ==="
    ip addr show
    echo

    echo "=== Load Average ==="
    uptime
    echo

    echo "=== Top 10 CPU Processes ==="
    ps aux --sort=-%cpu | head -11
    echo

    echo "=== Top 10 Memory Processes ==="
    ps aux --sort=-%mem | head -11
    echo

    echo "=== Network Connections ==="
    ss -s
    echo

    echo "=== System Limits ==="
    ulimit -a

} | tee "$OUTPUT_FILE"

echo
echo "Baseline saved to: $OUTPUT_FILE"
