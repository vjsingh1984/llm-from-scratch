#!/bin/bash
# Anomaly detector

CURRENT=$(get_metric)
BASELINE=$(cat /var/lib/baseline)

DEVIATION=$(echo "scale=2; ($CURRENT - $BASELINE) / $BASELINE * 100" | bc)

if (( $(echo "$DEVIATION > 20" | bc -l) )); then
    echo "Anomaly detected: ${DEVIATION}% deviation"
fi
