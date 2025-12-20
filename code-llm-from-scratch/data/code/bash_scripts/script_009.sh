#!/bin/bash
# Detect and handle zombie processes

echo "Zombie Process Hunter"
echo "===================="

# Find zombie processes
ZOMBIES=$(ps aux | awk '$8=="Z" {print $2}')

if [ -z "$ZOMBIES" ]; then
    echo "No zombie processes found"
    exit 0
fi

echo "Found zombie processes:"
ps aux | awk '$8=="Z" {print}'

echo
echo "Zombie PIDs: $ZOMBIES"

# For each zombie, try to clean up by signaling parent
for zpid in $ZOMBIES; do
    # Get parent PID
    PPID=$(ps -o ppid= -p "$zpid" 2>/dev/null | tr -d ' ')

    if [ -n "$PPID" ] && [ "$PPID" != "1" ]; then
        echo "Sending SIGCHLD to parent process $PPID"
        kill -CHLD "$PPID" 2>/dev/null

        sleep 1

        # Check if zombie still exists
        if ps -p "$zpid" > /dev/null 2>&1; then
            echo "Warning: Zombie $zpid still exists, parent may need restart"
            PARENT_CMD=$(ps -p "$PPID" -o comm= 2>/dev/null)
            echo "  Parent process: $PPID ($PARENT_CMD)"
        else
            echo "Successfully cleaned zombie $zpid"
        fi
    fi
done
