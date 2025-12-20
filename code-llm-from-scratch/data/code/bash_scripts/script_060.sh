#!/bin/bash
# Test network connectivity to multiple hosts

HOSTS=("8.8.8.8" "1.1.1.1" "google.com" "github.com")
TIMEOUT=5

echo "Network Connectivity Test"
echo "========================="
echo

test_ping() {
    local host=$1

    echo -n "Testing $host (ping)... "

    if ping -c 1 -W $TIMEOUT "$host" &>/dev/null; then
        echo "✓ OK"
        return 0
    else
        echo "✗ FAILED"
        return 1
    fi
}

test_dns() {
    local host=$1

    echo -n "Testing $host (DNS)... "

    if nslookup "$host" &>/dev/null; then
        echo "✓ OK"
        return 0
    else
        echo "✗ FAILED"
        return 1
    fi
}

test_http() {
    local host=$1

    echo -n "Testing $host (HTTP)... "

    if curl -s -o /dev/null -w "%{http_code}" --max-time $TIMEOUT "http://$host" | grep -q "^[23]"; then
        echo "✓ OK"
        return 0
    else
        echo "✗ FAILED"
        return 1
    fi
}

FAILURES=0

for host in "${HOSTS[@]}"; do
    test_ping "$host" || ((FAILURES++))
    test_dns "$host" || ((FAILURES++))
    test_http "$host" || ((FAILURES++))
    echo
done

echo "==================="
if [ $FAILURES -eq 0 ]; then
    echo "✓ All tests passed"
    exit 0
else
    echo "✗ $FAILURES test(s) failed"
    exit 1
fi
