#!/bin/bash
# Set and manage system resource limits for users/processes

USER="${1}"
LIMIT_TYPE="${2}"
LIMIT_VALUE="${3}"

usage() {
    echo "Usage: $0 <user|@group> <limit-type> <value>"
    echo
    echo "Limit types:"
    echo "  cpu      - CPU time (minutes)"
    echo "  mem      - Memory (MB)"
    echo "  nproc    - Number of processes"
    echo "  nofile   - Number of open files"
    echo "  fsize    - File size (MB)"
    echo
    echo "Examples:"
    echo "  $0 john cpu 60        # Limit john to 60 min CPU time"
    echo "  $0 @users mem 1024    # Limit users group to 1GB RAM"
    exit 1
}

if [ $# -lt 3 ]; then
    usage
fi

LIMITS_FILE="/etc/security/limits.conf"
BACKUP_FILE="/etc/security/limits.conf.backup-$(date +%Y%m%d)"

# Backup current limits
if [ ! -f "$BACKUP_FILE" ]; then
    cp "$LIMITS_FILE" "$BACKUP_FILE"
    echo "Backed up limits to: $BACKUP_FILE"
fi

set_limit() {
    local user=$1
    local type=$2
    local value=$3

    # Remove existing limits for this user/type
    sed -i "/$user.*$type/d" "$LIMITS_FILE"

    # Add new limits
    case $type in
        cpu)
            echo "$user hard cpu $value" >> "$LIMITS_FILE"
            echo "$user soft cpu $((value * 80 / 100))" >> "$LIMITS_FILE"
            ;;
        mem)
            # Convert MB to KB
            local kb=$((value * 1024))
            echo "$user hard rss $kb" >> "$LIMITS_FILE"
            echo "$user soft rss $((kb * 80 / 100))" >> "$LIMITS_FILE"
            ;;
        nproc)
            echo "$user hard nproc $value" >> "$LIMITS_FILE"
            echo "$user soft nproc $((value * 80 / 100))" >> "$LIMITS_FILE"
            ;;
        nofile)
            echo "$user hard nofile $value" >> "$LIMITS_FILE"
            echo "$user soft nofile $((value * 80 / 100))" >> "$LIMITS_FILE"
            ;;
        fsize)
            # Convert MB to KB
            local kb=$((value * 1024))
            echo "$user hard fsize $kb" >> "$LIMITS_FILE"
            echo "$user soft fsize $((kb * 80 / 100))" >> "$LIMITS_FILE"
            ;;
        *)
            echo "Error: Unknown limit type: $type"
            return 1
            ;;
    esac

    echo "✓ Set $type limit for $user to $value"
}

set_limit "$USER" "$LIMIT_TYPE" "$LIMIT_VALUE"

echo
echo "Current limits for $USER:"
grep "^$USER" "$LIMITS_FILE"

echo
echo "Note: Users need to log out and back in for changes to take effect"
