#!/bin/bash
add() {
    result=$(($1 + $2))
    echo $result
}

sum=$(add 5 3)
echo "Sum: $sum"
