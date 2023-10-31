#!/bin/bash

# FIRST SEQUENCE: ACTIVE FRACTION VALUES
# SECOND SEQUENCE: AVERAGE DEGREE VALUES
# THIRD SEQUENCE: THRESHOLD VALUES
# FOURTH SEQUENCE: VACCINATION RATE VALUES
# FIFTH SEQUENCE: ZEALOT FRACTION VALUES

#parallel --jobs 50 --progress --nice 10 /home/ademiguel/threshold/src/launch_threshold.sh $(seq 0 0.02 1.0) ::: $(seq 0 0.02 1.0)
parallel --jobs 100 --progress --nice 10 /home/ademiguel/threshold/src/launch_threshold.sh {} {} {} :::  $(seq 0 0.05 1.0) ::: 5 10 20 ::: $(seq 0 0.05 1.0) :::  0.001 0.005 0.01 0.05 1 ::: 0.0 0.01 0.05 0.1 0.25 0.5

