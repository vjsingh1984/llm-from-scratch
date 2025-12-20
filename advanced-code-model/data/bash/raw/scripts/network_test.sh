#!/bin/bash
# Network connectivity test

HOSTS=("google.com" "github.com" "stackoverflow.com")

echo "Network Connectivity Test"
echo "========================="
echo ""

for HOST in "${HOSTS[@]}"; do
    echo -n "Testing $HOST... "
    
    if ping -c 1 -W 2 "$HOST" > /dev/null 2>&1; then
        RESPONSE_TIME=$(ping -c 1 "$HOST" | grep 'time=' | awk -F'time=' '{print $2}' | awk '{print $1}')
        echo "✓ OK (${RESPONSE_TIME}ms)"
    else
        echo "✗ FAILED"
    fi
done

echo ""
echo "DNS Resolution:"
for HOST in "${HOSTS[@]}"; do
    echo "$HOST:"
    dig +short "$HOST" | head -3
done
