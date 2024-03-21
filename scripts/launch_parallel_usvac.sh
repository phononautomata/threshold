#!/bin/bash

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

parallel --jobs 4 --progress --nice 10 /Users/alfonso/workshop/threshold/src/launch_threshold_us.sh "massachusetts" {} ::: "${sequence_values[@]}"

