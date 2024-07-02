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

list_id_region=(
    Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware DistrictOfColumbia Florida 
    Georgia Hawaii Idaho Illinois Indiana Iowa Kansas Kentucky Louisiana Maine 
    Maryland Massachusetts Michigan Minnesota Mississippi Missouri Montana Nebraska Nevada NewHampshire 
    NewJersey NewMexico NewYork NorthCarolina NorthDakota Ohio Oklahoma Oregon Pennsylvania RhodeIsland 
    SouthCarolina SouthDakota Tennessee Texas Utah Vermont Virginia Washington WestVirginia Wisconsin 
    Wyoming National
)

size=100000

list_uuid=()

id_region_str=$(IFS=" "; echo "${list_id_region[*]}")

echo "Running Python script to collect UUIDs..."
python_output=$($PYTHON "${base_path}/${script_folder}/collect_networks.py" --id_region $id_region_str --size "$size")
echo "Python script output:"
echo "$python_output"

while IFS= read -r line; do
    list_uuid+=("$line")
done <<< "$python_output"

if [ ${#list_uuid[@]} -eq 0 ]; then
    echo "No UUIDs found for the given parameters."
    exit 1
fi

a=0.0001
b=1.0
n=41

list_rate_vaccination=()
for ((i=1; i<=n; i++)); do
  value=$(LC_NUMERIC=C awk -v a="$a" -v b="$b" -v i="$i" -v n="$n" 'BEGIN { printf "%.6f", a * (b/a)^((i-1)/(n-1)) }')
  list_rate_vaccination+=($value)
done

for uuid in "${list_uuid[@]}"; do
    for vaccination in "${list_rate_vaccination[@]}"; do
        echo "$uuid $vaccination"
    done
done | parallel --colsep ' ' --jobs 60 --progress --nice 10 "${base_path}/${script_folder}/launch_datadriven_threshold.sh" {1} {2}
