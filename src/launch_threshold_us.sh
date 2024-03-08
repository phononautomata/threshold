#!/bin/bash

STATE=$1
VACCINATION=$2

cargo run -r -- --usa-id $STATE --vaccination-rate $VACCINATION
