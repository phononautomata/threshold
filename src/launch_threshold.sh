#!/bin/bash
ACTIVE=$1
DEGREE=$2
THETA=$3
VACCINATION=$4
ZEALOT=$5

# Check if ACTIVE + ZEALOT is less than or equal to 1
if (( $(echo "$ACTIVE + $ZEALOT <= 1" | bc -l) )); then
    # If the condition is met, run the Rust program
    cargo run -r -- --active-fraction $ACTIVE --average-degree $DEGREE --threshold $THETA --vaccination-rate $VACCINATION --zealot-fraction $ZEALOT
else
    # If the condition is not met, skip execution
    echo "Skipping execution: ACTIVE + ZEALOT > 1"
fi
