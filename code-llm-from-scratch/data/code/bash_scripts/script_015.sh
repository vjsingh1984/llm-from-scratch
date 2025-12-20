#!/bin/bash
# Monitor network interface statistics

INTERFACE="${1:-eth0}"
INTERVAL="${2:-5}"

if ! ip link show "$INTERFACE" &>/dev/null; then
    echo "Error: Interface $INTERFACE not found"
    exit 1
fi

echo "Monitoring interface: $INTERFACE"
echo "Sample interval: ${INTERVAL}s"
echo "Press Ctrl+C to stop"
echo

get_stats() {
    local iface=$1
    local rx_bytes=$(cat "/sys/class/net/$iface/statistics/rx_bytes")
    local tx_bytes=$(cat "/sys/class/net/$iface/statistics/tx_bytes")
    local rx_packets=$(cat "/sys/class/net/$iface/statistics/rx_packets")
    local tx_packets=$(cat "/sys/class/net/$iface/statistics/tx_packets")
    local rx_errors=$(cat "/sys/class/net/$iface/statistics/rx_errors")
    local tx_errors=$(cat "/sys/class/net/$iface/statistics/tx_errors")

    echo "$rx_bytes $tx_bytes $rx_packets $tx_packets $rx_errors $tx_errors"
}

format_bytes() {
    local bytes=$1
    if [ $bytes -gt 1073741824 ]; then
        echo "$(awk "BEGIN {printf "%.2f", $bytes/1073741824}") GB/s"
    elif [ $bytes -gt 1048576 ]; then
        echo "$(awk "BEGIN {printf "%.2f", $bytes/1048576}") MB/s"
    elif [ $bytes -gt 1024 ]; then
        echo "$(awk "BEGIN {printf "%.2f", $bytes/1024}") KB/s"
    else
        echo "$bytes B/s"
    fi
}

# Initial reading
read rx_bytes_old tx_bytes_old rx_pkts_old tx_pkts_old rx_err_old tx_err_old <<< $(get_stats "$INTERFACE")

while true; do
    sleep "$INTERVAL"

    read rx_bytes_new tx_bytes_new rx_pkts_new tx_pkts_new rx_err_new tx_err_new <<< $(get_stats "$INTERFACE")

    # Calculate deltas
    rx_rate=$(( (rx_bytes_new - rx_bytes_old) / INTERVAL ))
    tx_rate=$(( (tx_bytes_new - tx_bytes_old) / INTERVAL ))
    rx_pps=$(( (rx_pkts_new - rx_pkts_old) / INTERVAL ))
    tx_pps=$(( (tx_pkts_new - tx_pkts_old) / INTERVAL ))

    echo "$(date '+%H:%M:%S') | RX: $(format_bytes $rx_rate) ($rx_pps pps) | TX: $(format_bytes $tx_rate) ($tx_pps pps) | Errors: RX=$rx_err_new TX=$tx_err_new"

    # Update old values
    rx_bytes_old=$rx_bytes_new
    tx_bytes_old=$tx_bytes_new
    rx_pkts_old=$rx_pkts_new
    tx_pkts_old=$tx_pkts_new
done
