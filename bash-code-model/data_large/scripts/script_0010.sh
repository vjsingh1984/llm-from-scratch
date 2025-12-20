#!/bin/bash
# Simple process manager

PROCESS_NAME=$1
ACTION=$2
PID_FILE="/var/run/${PROCESS_NAME}.pid"

start_process() {
    if [ -f "$PID_FILE" ]; then
        echo "Process already running (PID: $(cat $PID_FILE))"
        return 1
    fi

    echo "Starting $PROCESS_NAME..."
    nohup /usr/bin/$PROCESS_NAME > /dev/null 2>&1 &
    echo $! > "$PID_FILE"
    echo "Started with PID: $(cat $PID_FILE)"
}

stop_process() {
    if [ ! -f "$PID_FILE" ]; then
        echo "Process not running"
        return 1
    fi

    PID=$(cat "$PID_FILE")
    echo "Stopping $PROCESS_NAME (PID: $PID)..."
    kill "$PID"

    # Wait for process to stop
    timeout=10
    while kill -0 "$PID" 2>/dev/null && [ $timeout -gt 0 ]; do
        sleep 1
        timeout=$((timeout - 1))
    done

    if kill -0 "$PID" 2>/dev/null; then
        echo "Force killing process..."
        kill -9 "$PID"
    fi

    rm -f "$PID_FILE"
    echo "Stopped"
}

status_process() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "$PROCESS_NAME is running (PID: $PID)"
            return 0
        else
            echo "$PROCESS_NAME is not running (stale PID file)"
            return 1
        fi
    else
        echo "$PROCESS_NAME is not running"
        return 1
    fi
}

case "$ACTION" in
    start)
        start_process
        ;;
    stop)
        stop_process
        ;;
    restart)
        stop_process
        sleep 2
        start_process
        ;;
    status)
        status_process
        ;;
    *)
        echo "Usage: $0 <process-name> {start|stop|restart|status}"
        exit 1
        ;;
esac
