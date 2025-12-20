#!/bin/bash
# Monitor network bandwidth usage

INTERFACE="${1:-eth0}"
INTERVAL="${2:-1}"

echo "Monitoring bandwidth on $INTERFACE (${INTERVAL}s intervals)"
echo "Press Ctrl+C to stop"
echo

get_bytes() {
    cat "/sys/class/net/$INTERFACE/statistics/$1_bytes"
}

format_bps() {
    local bps=$1
    if [ $bps -gt 1073741824 ]; then
        echo "$(awk "BEGIN {printf "%.2f", $bps/1073741824}") Gbps"
    elif [ $bps -gt 1048576 ]; then
        echo "$(awk "BEGIN {printf "%.2f", $bps/1048576}") Mbps"
    elif [ $bps -gt 1024 ]; then
        echo "$(awk "BEGIN {printf "%.2f", $bps/1024}") Kbps"
    else
        echo "$bps bps"
    fi
}

rx_old=$(get_bytes rx)
tx_old=$(get_bytes tx)

while true; do
    sleep "$INTERVAL"

    rx_new=$(get_bytes rx)
    tx_new=$(get_bytes tx)

    rx_bps=$(( (rx_new - rx_old) * 8 / INTERVAL ))
    tx_bps=$(( (tx_new - tx_old) * 8 / INTERVAL ))

    echo "$(date '+%H:%M:%S') | RX: $(format_bps $rx_bps) | TX: $(format_bps $tx_bps)"

    rx_old=$rx_new
    tx_old=$tx_new
done
