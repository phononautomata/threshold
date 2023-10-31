#!/bin/bash

# Set the source and destination paths
dest_dir="/Users/ademiguel/workshop/threshold/"
source_dir="ademiguelr@srv-87.unizar.bifi.es:/home/ademiguel/threshold/results"

# Use scp to copy the directory and its contents
scp -r "$source_dir" "$dest_dir"
