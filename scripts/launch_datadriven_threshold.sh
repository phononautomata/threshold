#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 REGION SIZE UUID VACCINATION"
    exit 1
fi

cd "$(dirname "$0")"/..

OPINION="data-driven-thresholds"
REGION=$1
SIZE=$2
UUID=$3
VACCINATION=$4

cargo run -r -- --id-experiment 2 --model-opinon $OPINION --model-region $REGION --nagents $SIZE --uuid-multilayer $UUID --rate-vaccination $VACCINATION
