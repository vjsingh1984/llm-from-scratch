#!/bin/bash
# Check and enforce file system quotas

QUOTA_LIMIT_GB=100
WARNING_THRESHOLD=80  # percentage

check_user_quota() {
    local user=$1

    # Get user's home directory usage
    local usage=$(du -sh "/home/$user" 2>/dev/null | awk '{print $1}')
    local usage_gb=$(du -sb "/home/$user" 2>/dev/null | awk '{print int($1/1024/1024/1024)}')

    echo "User: $user"
    echo "  Usage: $usage ($usage_gb GB)"

    # Check if over quota
    if [ "$usage_gb" -gt "$QUOTA_LIMIT_GB" ]; then
        echo "  STATUS: OVER QUOTA"
        echo "User $user is over quota: ${usage_gb}GB / ${QUOTA_LIMIT_GB}GB" |             mail -s "Quota Alert" "${user}@localhost"
        return 1
    fi

    # Check if approaching quota
    local percent=$(( usage_gb * 100 / QUOTA_LIMIT_GB ))
    if [ "$percent" -gt "$WARNING_THRESHOLD" ]; then
        echo "  STATUS: WARNING (${percent}% used)"
        echo "You are using ${percent}% of your quota" |             mail -s "Quota Warning" "${user}@localhost"
    else
        echo "  STATUS: OK (${percent}% used)"
    fi

    return 0
}

echo "File System Quota Check"
echo "======================="
echo "Quota Limit: ${QUOTA_LIMIT_GB}GB"
echo

# Check all regular users
for user in $(getent passwd | awk -F: '$3 >= 1000 && $3 < 65000 {print $1}'); do
    if [ -d "/home/$user" ]; then
        check_user_quota "$user"
        echo
    fi
done
