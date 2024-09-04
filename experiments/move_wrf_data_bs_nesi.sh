#!/bin/bash

# Base directory path
BASE_DIR="/mnt/deepweather_data/metservice/2024"

# Destination server details
DEST_SERVER="mahuika"
DEST_PATH="//nesi/nobackup/nesi03947/deepweather_data/2024"

## SCP only files in the nz4kmN-ECMWF-SIGMA/ directories
for month in {01..12}; do
    # Construct the month directory path
    MONTH_DIR="$BASE_DIR/$month"

    # Check if the month directory exists
    if [ -d "$MONTH_DIR" ]; then
        echo "Processing month: $month"

        for sub_dir in "$MONTH_DIR"/*/nz4kmN-ECMWF-SIGMA/; do
            # Check if the directory exists
            if [ -d "$sub_dir" ]; then
                echo "Copying files from $sub_dir"

                # SCP all files in the nz4kmN-ECMWF-SIGMA/ directory
                scp "$sub_dir"* "$DEST_USER@$DEST_SERVER:$DEST_PATH/$month/"
            fi
        done

        echo "Finished processing month: $month"
    else
        echo "Month directory $month does not exist. Skipping."
    fi
done

echo "All files have been processed."