#!/bin/bash

PYTHON=$(which python3)
if [ ! -x "$PYTHON" ]; then
    echo "Python interpreter not found."
    exit 1
fi

echo "Using Python at $PYTHON"
echo "Python path: $($PYTHON -c 'import sys; print(sys.path)')"

base_path=$(cd "$(dirname "$0")"/.. && pwd)
script_folder="scripts"

id_region="National"
size=100000
storage_path="/data/alfonso/threshold/data/curated/networks"

list_uuid=()

echo "Running Python script to collect UUIDs..."
python_output=$($PYTHON "${base_path}/${script_folder}/collect_networks.py" --id_region "$id_region" --size "$size")
echo "Python script output:"
echo "$python_output"

while IFS= read -r line; do
    list_uuid+=("$line")
done <<< "$python_output"

if [ ${#list_uuid[@]} -eq 0 ]; then
    echo "No UUIDs found for the given parameters."
    exit 1
fi

list_fraction_active=$(seq 0 0.05 1)
list_threshold_opinion=$(seq 0 0.05 1)
list_rate_vaccination="0.001 0.005 0.01 0.05"
list_fraction_zealot="0.0"

model_hesitancy="random"

for uuid in "${list_uuid[@]}"; do
    for active in $list_fraction_active; do
        for threshold in $list_threshold_opinion; do
            for vaccination in $list_rate_vaccination; do
                for zealot in $list_fraction_zealot; do
                    echo "$uuid $model_hesitancy $active $threshold $vaccination $zealot"
                done
            done
        done
    done
done | parallel --colsep ' ' --jobs 60 --progress --nice 10 "${base_path}/${script_folder}/launch_homogeneous_threshold.sh" {1} {2} {3} {4} {5} {6}