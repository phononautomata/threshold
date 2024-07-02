#!/bin/bash

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 UUID HESITANCY ACTIVE THRESHOLD VACCINATION ZEALOT"
    exit 1
fi

cd "$(dirname "$0")"/..


OPINION="homogeneous-thresholds"

UUID=$1
HESITANCY=$2
ACTIVE=$3
THRESHOLD=$4
VACCINATION=$5
ZEALOT=$6

cargo run -r -- --id-experiment 2 --model-hesitancy $HESITANCY --model-opinion $OPINION --fraction-active $ACTIVE --fraction-zealot $ZEALOT --rate-vaccination $VACCINATION --string-multilayer $UUID --threshold-opinion $THRESHOLD
