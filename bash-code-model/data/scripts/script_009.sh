#!/bin/bash
if [ ! -d "backup" ]; then
    mkdir backup
fi
cp file.txt backup/
echo "Backup created"
