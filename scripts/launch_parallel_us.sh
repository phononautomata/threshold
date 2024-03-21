#!/bin/bash

# Call generate_states_list.sh to generate a comma-separated list of all 50 US states
states_list="$(./src/generate_states_list.sh)"

a=0.0001
b=1.0
n=41

sequence_values=()
for ((i=1; i<=n; i++)); do
  value=$(LC_NUMERIC=C awk -v a="$a" -v b="$b" -v i="$i" -v n="$n" 'BEGIN { printf "%.6f", a * (b/a)^((i-1)/(n-1)) }')
  sequence_values+=($value)
done

# Print the generated values
echo "Generated sequence values: ${sequence_values[@]}"

# Call launch_threshold_us.sh for each state in parallel
for state in $(echo "$states_list" | tr ',' '\n')
do
  #parallel --jobs 7 --progress --nice 10 /Users/alfonso/workshop/threshold/src/launch_threshold_us.sh "$state" {} ::: "${sequence_values[@]}"
  parallel --jobs 7 --progress --nice 10 /Users/alfonso/workshop/threshold/src/launch_threshold_us.sh "$state" {} {} ::: 0.005 0.01 0.05 ::: 0.05 0.1 0.15
done
