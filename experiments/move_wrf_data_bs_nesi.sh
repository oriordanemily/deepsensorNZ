#!/bin/bash

YEAR="2023"

# Base directory path
BASE_DIR="/mnt/deepweather_data/metservice/$YEAR"

# Destination server details
DEST_SERVER="mahuika"
DEST_PATH="//nesi/nobackup/nesi03947/deepweather_data/$YEAR"

## SCP only files in the nz4kmN-ECMWF-SIGMA/ directories
for month in {09..12}; do
    # Construct the month directory path
    MONTH_DIR="$BASE_DIR/$month"

    # Check if the month directory exists
    if [ -d "$MONTH_DIR" ]; then
        echo "Processing month: $month"

        for sub_dir in "$MONTH_DIR"/*/nz4kmN-ECMWF-SIGMA; do
            # Check if the directory exists
            if [ -d "$sub_dir" ]; then
                echo "Copying files from $sub_dir"

                # Get the final foldername from sub_dir
                foldername=$(basename "$(dirname "$sub_dir")")

                # SCP all files in the nz4kmN-ECMWF-SIGMA/ directory
                scp "$sub_dir"/*d02*00 "$DEST_SERVER:$DEST_PATH/$foldername/" # only copying the midnight runs for now
            fi
        done

        echo "Finished processing month: $month"
    else
        echo "Month directory $month does not exist. Skipping."
    fi
done

echo "All files have been processed."