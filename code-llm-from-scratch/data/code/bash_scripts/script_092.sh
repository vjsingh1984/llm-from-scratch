#!/bin/bash
# A/B test deployment

VARIANT_A="current"
VARIANT_B="${1}"

echo "Starting A/B test: $VARIANT_A vs $VARIANT_B"

deploy_variant "$VARIANT_B"
configure_traffic_split 50 50

echo "✓ A/B test configured"
