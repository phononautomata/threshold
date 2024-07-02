#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 UUID VACCINATION"
    exit 1
fi

cd "$(dirname "$0")"/..

HESITANCY="random"
OPINION="data-driven-thresholds"

UUID=$1
VACCINATION=$2

cargo run -r -- --id-experiment 2 --model-hesitancy $HESITANCY --model-opinion $OPINION --string-multilayer $UUID --rate-vaccination $VACCINATION