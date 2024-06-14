#!/bin/bash

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 REGION ACTIVE THRESHOLD VACCINATION ZEALOT"
    exit 1
fi

cd "$(dirname "$0")"/..

OPINION="homogeneous"

REGION=$1
ACTIVE=$2
THRESHOLD=$3
VACCINATION=$4
ZEALOT=$5

cargo run -r -- --id-experiment 2 --model-opinion $OPINION --model-region $REGION --fraction-active $ACTIVE --fraction-zealot $ZEALOT --rate-vaccination $VACCINATION --threshold-opinion $THRESHOLD
