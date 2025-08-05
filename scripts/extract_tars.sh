#!/bin/bash

# Directory where your tar files live
tar_dir="/leonardo_work/FUAL8_UKAEA_ML/apolansk/al4pde/data/data/jorek/tars"
# Pool directory
pool_dir="/leonardo_work/FUAL8_UKAEA_ML/apolansk/al4pde/data/data/jorek/pool"


# Loop through all tar files
for tar_file in "$tar_dir"/*.tar; do
    echo "Extracting $tar_file"
    tar -xf "$tar_file" -C "$pool_dir"
done

