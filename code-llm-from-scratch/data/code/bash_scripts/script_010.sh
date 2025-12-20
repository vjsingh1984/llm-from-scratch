#!/bin/bash
# Validate and test cron jobs

CRON_DIR="/var/spool/cron/crontabs"
TEST_MODE="${1:-check}"

validate_cron_syntax() {
    local cronline=$1

    # Skip comments and empty lines
    if [[ "$cronline" =~ ^#.*$ ]] || [ -z "$cronline" ]; then
        return 0
    fi

    # Basic validation
    if [[ ! "$cronline" =~ ^[0-9\*,\-/]+ ]]; then
        echo "  ✗ Invalid syntax: $cronline"
        return 1
    fi

    echo "  ✓ Valid syntax: $cronline"
    return 0
}

check_command_exists() {
    local cronline=$1
    local cmd=$(echo "$cronline" | awk '{for(i=6;i<=NF;i++) printf "%s ", $i}' | awk '{print $1}')

    if [ -n "$cmd" ] && ! command -v "$cmd" &> /dev/null; then
        echo "  ⚠ Command not found: $cmd"
        return 1
    fi

    return 0
}

validate_user_crontab() {
    local user=$1
    local cronfile="$CRON_DIR/$user"

    if [ ! -f "$cronfile" ]; then
        return 0
    fi

    echo "Validating crontab for user: $user"
    echo "=================================="

    local issues=0

    while IFS= read -r line; do
        validate_cron_syntax "$line" || ((issues++))
        check_command_exists "$line" || ((issues++))
    done < "$cronfile"

    if [ $issues -eq 0 ]; then
        echo "✓ No issues found"
    else
        echo "✗ Found $issues issue(s)"
    fi

    echo
    return $issues
}

echo "Cron Job Validator"
echo "=================="
echo

total_issues=0

# Check system crontab
if [ -f /etc/crontab ]; then
    echo "Checking /etc/crontab"
    validate_cron_syntax "$(cat /etc/crontab)"
fi

# Check user crontabs
for user in $(ls "$CRON_DIR" 2>/dev/null); do
    validate_user_crontab "$user"
    total_issues=$((total_issues + $?))
done

if [ $total_issues -gt 0 ]; then
    echo "Total issues found: $total_issues"
    exit 1
else
    echo "All cron jobs validated successfully"
    exit 0
fi
