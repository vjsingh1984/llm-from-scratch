#!/bin/bash
# Kubernetes deployment script

NAMESPACE="${1:-default}"
DEPLOYMENT="${2}"
IMAGE="${3}"

if [ -z "$DEPLOYMENT" ] || [ -z "$IMAGE" ]; then
    echo "Usage: $0 <namespace> <deployment> <image>"
    exit 1
fi

echo "Deploying to Kubernetes"
echo "======================="
echo "Namespace: $NAMESPACE"
echo "Deployment: $DEPLOYMENT"
echo "Image: $IMAGE"
echo

# Check if deployment exists
if ! kubectl get deployment "$DEPLOYMENT" -n "$NAMESPACE" &>/dev/null; then
    echo "Error: Deployment $DEPLOYMENT not found in namespace $NAMESPACE"
    exit 1
fi

# Update deployment
echo "Updating deployment..."
kubectl set image "deployment/$DEPLOYMENT"     "$DEPLOYMENT=$IMAGE"     -n "$NAMESPACE"

# Wait for rollout
echo "Waiting for rollout to complete..."
kubectl rollout status "deployment/$DEPLOYMENT" -n "$NAMESPACE"

# Check status
if [ $? -eq 0 ]; then
    echo "✓ Deployment successful"

    # Show new pods
    echo
    echo "New pods:"
    kubectl get pods -n "$NAMESPACE" -l "app=$DEPLOYMENT"
else
    echo "✗ Deployment failed"

    # Rollback
    echo "Rolling back..."
    kubectl rollout undo "deployment/$DEPLOYMENT" -n "$NAMESPACE"

    exit 1
fi
