#!/bin/bash
# Auto-scaling trigger

METRIC="${1:-cpu}"
THRESHOLD="${2:-80}"

echo "Monitoring $METRIC for auto-scaling (threshold: $THRESHOLD%)"

# Monitor and scale
current_value=$(get_metric_value "$METRIC")

if [ "$current_value" -gt "$THRESHOLD" ]; then
    echo "Scaling up..."
    kubectl scale deployment/app --replicas=$((current_replicas + 1))
fi
