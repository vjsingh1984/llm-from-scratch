#!/bin/bash
# TCP connection monitor

PORT="${1:-80}"

echo "Monitoring TCP connections on port $PORT..."

watch -n 1 "ss -tan | grep :$PORT | wc -l"
