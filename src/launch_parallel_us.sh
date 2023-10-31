#!/bin/bash

# Call generate_states_list.sh to generate a comma-separated list of all 50 US states
states_list="$(./generate_states_list.sh)"

# Call launch_threshold_us.sh for each state in parallel
for state in $(echo "$states_list" | tr ',' '\n')
do
  parallel --jobs 50 --progress --nice 10 /home/ademiguel/threshold/src/launch_threshold_us.sh "$state"
done
