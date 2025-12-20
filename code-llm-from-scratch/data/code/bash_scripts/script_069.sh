#!/bin/bash
# Bandwidth limiter

INTERFACE="${1}"
LIMIT="${2:-1mbit}"

tc qdisc add dev "$INTERFACE" root tbf rate "$LIMIT" burst 32kbit latency 400ms

echo "✓ Bandwidth limited to $LIMIT on $INTERFACE"
