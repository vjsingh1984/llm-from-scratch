#!/bin/bash
# Rollback automation

DEPLOYMENT="${1}"
NAMESPACE="${2:-default}"

echo "Rolling back deployment: $DEPLOYMENT"

kubectl rollout undo "deployment/$DEPLOYMENT" -n "$NAMESPACE"
kubectl rollout status "deployment/$DEPLOYMENT" -n "$NAMESPACE"

if [ $? -eq 0 ]; then
    echo "✓ Rollback successful"
else
    echo "✗ Rollback failed"
    exit 1
fi
