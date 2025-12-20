#!/bin/bash
trap "echo 'Interrupted'; exit" INT TERM
while true; do
    sleep 1
done
