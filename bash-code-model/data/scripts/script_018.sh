#!/bin/bash
if [ -z "$VAR" ]; then
    echo "VAR is empty"
fi

if [ $# -eq 0 ]; then
    echo "No arguments"
fi
