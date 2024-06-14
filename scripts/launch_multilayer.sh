#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 REGION SIZE"
    exit 1
fi

cd "$(dirname "$0")"/..

REGION=$1
SIZE=$2

cargo run -r -- --id-experiment 1 --model-region $REGION --nagents $SIZE