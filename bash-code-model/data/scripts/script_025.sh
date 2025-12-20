#!/bin/bash
if [ -r "file.txt" ]; then
    echo "Readable"
fi

if [ -w "file.txt" ]; then
    echo "Writable"
fi

if [ -x "script.sh" ]; then
    echo "Executable"
fi
