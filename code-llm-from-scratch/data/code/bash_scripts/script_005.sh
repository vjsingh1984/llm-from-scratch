#!/bin/bash
# Automated system update with safety checks

set -euo pipefail

LOG_FILE="/var/log/system-update.log"
REBOOT_REQUIRED="/var/run/reboot-required"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

check_disk_space() {
    local available=$(df / | awk 'NR==2 {print $4}')
    local required=1048576  # 1GB in KB

    if [ "$available" -lt "$required" ]; then
        log "ERROR: Insufficient disk space"
        exit 1
    fi
}

backup_package_list() {
    log "Backing up package list..."
    dpkg --get-selections > /backup/package-list-$(date +%Y%m%d).txt
}

perform_update() {
    log "Starting system update..."

    # Update package lists
    apt-get update

    # Upgrade packages
    apt-get upgrade -y

    # Clean up
    apt-get autoremove -y
    apt-get autoclean

    log "Update complete"
}

check_reboot() {
    if [ -f "$REBOOT_REQUIRED" ]; then
        log "NOTICE: System reboot required"

        cat "$REBOOT_REQUIRED"

        # Schedule reboot for 2 AM
        echo "shutdown -r 02:00 'Scheduled reboot for updates'" | at now
    fi
}

main() {
    log "=== System Update Started ==="

    check_disk_space
    backup_package_list
    perform_update
    check_reboot

    log "=== System Update Completed ==="
}

main "$@"
