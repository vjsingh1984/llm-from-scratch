#!/bin/bash
set -e
command_that_might_fail || {
    echo "Command failed"
    exit 1
}
echo "Success"
