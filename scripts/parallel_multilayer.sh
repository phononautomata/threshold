#!/bin/bash

#base_path="/Users/alfonso/workshop/threshold"
base_path="$HOME/threshold"
script_folder="scripts"

if [ -z "$base_path" ]; then
    echo "The THRESHOLD_PATH environment variable is not set. Please set it to the base path of the expidemics project."
    exit 1
fi

list_model_region=(
    alabama
    alaska
    arizona
    arkansas
    california
    colorado
    connecticut
    delaware
    florida
    georgia
    hawaii
    idaho
    illinois
    indiana
    iowa
    kansas
    kentucky
    louisiana
    maine
    maryland
    massachusetts
    michigan
    minnesota
    mississippi
    missouri
    montana
    nebraska
    nevada
    new-hampshire
    new-jersey
    new-mexico
    new-york
    north-carolina
    north-dakota
    ohio
    oklahoma
    oregon
    pennsylvania
    rhode-island
    south-carolina
    south-dakota
    tennessee
    texas
    utah
    vermont
    virginia
    washington
    west-virginia
    wisconsin
    wyoming
    national
)

size=100000
n_repeats=25

for region in "${list_model_region[@]}"; do
    for ((i=1; i<=n_repeats; i++)); do
        echo "$region $size"
    done
done | parallel --colsep ' ' --jobs 25 --progress --nice 10 "${base_path}/${script_folder}/launch_multilayer.sh" {1} {2}