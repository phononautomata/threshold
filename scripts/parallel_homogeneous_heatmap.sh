#!/bin/bash

VENV_PATH="/home/alfonso/threshold/.threshold"

if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
    echo "Activated virtual environment."
    echo "Using Python at $(which python3)"
else
    echo "Virtual environment not found."
    exit 1
fi

PYTHON="${VENV_PATH}/bin/python3"
echo "Python path from virtual environment: $($PYTHON -c 'import sys; print(sys.path)')"

base_path=$(cd "$(dirname "$0")"/.. && pwd)
script_folder="scripts"

list_uuid=()

id_region="national"
size=100000
storage_path=true

# Collecting network UUIDs
while IFS= read -r line; do
    list_uuid+=("$line")
done < <($PYTHON "${base_path}/${script_folder}/collect_networks.py" --id_region "$id_region" --size "$size" --path_storage "$storage_path")

if [ ${#list_uuid[@]} -eq 0 ]; then
    echo "No UUIDs found for the given parameters."
    exit 1
fi

list_fraction_active=$(seq 0 0.05 1)
list_threshold_opinion=$(seq 0 0.05 1)
list_rate_vaccination="0.001 0.005 0.01 0.05"
list_fraction_zealot="0.0"

model_region="national"

# Iterate over combinations of parameters and run jobs in parallel
for uuid in "${list_uuid[@]}"; do
    for active in $list_fraction_active; do
        for threshold in $list_threshold_opinion; do
            for vaccination in $list_rate_vaccination; do
                for zealot in $list_fraction_zealot; do
                    echo "$model_region $uuid $active $threshold $vaccination $zealot"
                done
            done
        done
    done
done | parallel --colsep ' ' --jobs 60 --progress --nice 10 "${base_path}/${script_folder}/launch_homogeneous_threshold.sh" {1} {2} {3} {4} {5} {6}"