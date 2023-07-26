#!/bin/bash

process_subdirectories() {
    local parent_dir="$1"

    for subdirectory in "$parent_dir"/*; do
        if [ -d "$subdirectory" ]; then
            dir_name=$(basename "$subdirectory")
            eval "python e2k.py -d $parent_dir/$dir_name/"
        fi
    done
}

volcanic_test_dir="volcanic_test"
test_cases_dir="test_cases"

process_subdirectories "$volcanic_test_dir" 
process_subdirectories "$test_cases_dir" 
