#!/bin/bash
# Check CI/CD pipeline status

PIPELINE_ID="${1}"
API_URL="${CI_API_URL:-https://ci.example.com/api}"
API_TOKEN="${CI_API_TOKEN}"

if [ -z "$PIPELINE_ID" ]; then
    echo "Usage: $0 <pipeline-id>"
    exit 1
fi

get_pipeline_status() {
    curl -s -H "Authorization: Bearer $API_TOKEN"         "$API_URL/pipelines/$PIPELINE_ID"
}

wait_for_completion() {
    echo "Waiting for pipeline $PIPELINE_ID to complete..."

    while true; do
        response=$(get_pipeline_status)
        status=$(echo "$response" | jq -r '.status')

        echo -n "."

        case $status in
            success)
                echo
                echo "✓ Pipeline succeeded"
                return 0
                ;;
            failed)
                echo
                echo "✗ Pipeline failed"
                echo "$response" | jq -r '.stages[] | select(.status=="failed") | "  Failed stage: \(.name)"'
                return 1
                ;;
            running|pending)
                sleep 10
                ;;
            *)
                echo
                echo "Unknown status: $status"
                return 1
                ;;
        esac
    done
}

echo "Pipeline Status Check"
echo "===================="
echo "Pipeline ID: $PIPELINE_ID"
echo

wait_for_completion
exit $?
