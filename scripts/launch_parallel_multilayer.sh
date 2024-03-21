#!/bin/bash

# Define the states as an array
states=(
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

# Define the vaccination rates
vaccination_rates="0.005 0.01 0.05"

# Generate the combination of arguments for GNU Parallel
for state in "${states[@]}"; do
    for rate in "${vaccination_rates[@]}"; do
        echo "$state $rate"
    done
done | parallel --colsep ' ' --jobs 60 --progress --nice 10 /Users/alfonso/workshop/threshold/src/launch_threshold_us.sh {1} {2}
