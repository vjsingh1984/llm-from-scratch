#!/bin/bash
# Threshold alert system

METRIC_VALUE=$(get_metric_value)
THRESHOLD=80

if [ "$METRIC_VALUE" -gt "$THRESHOLD" ]; then
    echo "Alert: Metric exceeded threshold"
fi
