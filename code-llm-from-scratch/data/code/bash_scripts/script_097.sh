#!/bin/bash
# Service mesh deployment

SERVICE="${1}"

kubectl apply -f "k8s/service-mesh/$SERVICE.yaml"

echo "✓ Service mesh configuration applied"
