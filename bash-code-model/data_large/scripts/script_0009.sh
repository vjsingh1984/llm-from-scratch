#!/bin/bash
# API endpoint health checker

ENDPOINTS=(
    "https://api.example.com/health"
    "https://api.example.com/status"
    "https://api.example.com/version"
)

TIMEOUT=10
REPORT_FILE="api_health_$(date +%Y%m%d_%H%M%S).txt"

{
    echo "API Health Check Report"
    echo "======================="
    echo "Date: $(date)"
    echo

    for endpoint in "${ENDPOINTS[@]}"; do
        echo "Testing: $endpoint"

        start_time=$(date +%s%N)
        response=$(curl -s -w "HTTP_CODE:%{http_code}" --max-time $TIMEOUT "$endpoint")
        end_time=$(date +%s%N)

        http_code=$(echo "$response" | grep -o "HTTP_CODE:[0-9]*" | cut -d':' -f2)
        response_time=$(( (end_time - start_time) / 1000000 ))

        if [ "$http_code" = "200" ]; then
            echo "  ✓ Status: OK"
        else
            echo "  ✗ Status: Failed (HTTP $http_code)"
        fi

        echo "  Response time: ${response_time}ms"
        echo
    done

    echo "Report complete"
} | tee "$REPORT_FILE"
