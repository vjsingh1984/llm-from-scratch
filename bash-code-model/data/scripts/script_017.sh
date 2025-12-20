#!/bin/bash
echo "Log message" >> log.txt
command 2>&1 | tee output.log
cat < input.txt > output.txt
