#!/bin/bash

# Define an array with the names of all 50 US states
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

# Convert the array to a comma-separated list
IFS=','; states_list="${states[*]}"; unset IFS

echo "$states_list"
