#!/bin/bash
text="Hello World"
echo "${text,,}"  # lowercase
echo "${text^^}"  # uppercase
echo "${text:0:5}"  # substring
