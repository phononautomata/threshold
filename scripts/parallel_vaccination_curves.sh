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

list_model_region=(
    alabama alaska arizona arkansas california colorado connecticut delaware florida georgia
    hawaii idaho illinois indiana iowa kansas kentucky louisiana maine maryland massachusetts
    michigan minnesota mississippi missouri montana nebraska nevada new-hampshire new-jersey
    new-mexico new-york north-carolina north-dakota ohio oklahoma oregon pennsylvania
    rhode-island south-carolina south-dakota tennessee texas utah vermont virginia washington
    west-virginia wisconsin wyoming national
)

a=0.0001
b=1.0
n=41

list_rate_vaccination=()
for ((i=1; i<=n; i++)); do
  value=$(LC_NUMERIC=C awk -v a="$a" -v b="$b" -v i="$i" -v n="$n" 'BEGIN { printf "%.6f", a * (b/a)^((i-1)/(n-1)) }')
  list_rate_vaccination+=($value)
done

# Removed the second initialization of list_uuid
# Iterate over combinations of parameters and run jobs in parallel
for region in "${list_model_region[@]}"; do
    for uuid in "${list_uuid[@]}"; do
        for vaccination in "${list_rate_vaccination[@]}"; do
            echo "$region $size $uuid $vaccination"
        done
    done
done | parallel --colsep ' ' --jobs 60 --progress --nice 10 "${base_path}/${script_folder}/launch_datadriven_threshold.sh" {1} {2} {3} {4}
