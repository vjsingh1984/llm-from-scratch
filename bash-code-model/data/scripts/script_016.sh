#!/bin/bash
cat file.txt | grep "error" | wc -l
ps aux | grep python | awk '{print $2}'
