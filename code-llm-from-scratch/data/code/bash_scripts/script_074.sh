#!/bin/bash
# Collect system performance metrics

METRICS_FILE="/var/log/metrics/$(date +%Y%m%d_%H%M%S).json"

mkdir -p "$(dirname "$METRICS_FILE")"

{
    echo "{"
    echo "  "timestamp": "$(date -Iseconds)","
    echo "  "hostname": "$(hostname)","
    echo "  "cpu": {"
    echo "    "usage": $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1),"
    echo "    "cores": $(nproc),"
    echo "    "load_avg": "$(uptime | awk -F'load average:' '{print $2}' | xargs)""
    echo "  },"
    echo "  "memory": {"
    echo "    "total_mb": $(free -m | awk 'NR==2{print $2}'),"
    echo "    "used_mb": $(free -m | awk 'NR==2{print $3}'),"
    echo "    "free_mb": $(free -m | awk 'NR==2{print $4}'),"
    echo "    "usage_percent": $(free | grep Mem | awk '{printf("%.1f", ($3/$2) * 100)}')"
    echo "  },"
    echo "  "disk": {"
    echo "    "usage_percent": $(df -h / | awk 'NR==2{print $5}' | sed 's/%//'),"
    echo "    "available_gb": $(df -BG / | awk 'NR==2{print $4}' | sed 's/G//')"
    echo "  },"
    echo "  "network": {"
    echo "    "connections": $(ss -tan | wc -l)"
    echo "  }"
    echo "}"
} > "$METRICS_FILE"

echo "✓ Metrics collected: $METRICS_FILE"
