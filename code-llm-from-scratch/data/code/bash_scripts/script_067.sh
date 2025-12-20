#!/bin/bash
# Network packet capture

INTERFACE="${1:-eth0}"
COUNT="${2:-100}"

tcpdump -i "$INTERFACE" -c "$COUNT" -w "capture-$(date +%Y%m%d_%H%M%S).pcap"
