#!/bin/bash

FMRIPREP_OUTPUT_RIA="/cbica/projects/executive_function/mebold_trt/derivatives/nordic_fmriprep_babs_project/output_ria"
ZIP_DIR="/cbica/projects/executive_function/mebold_trt/derivatives/nordic_fmriprep_zipped_ephe"
UNZIP_DIR="/cbica/projects/executive_function/mebold_trt/derivatives/nordic_fmriprep_unzipped"

mkdir -p "$UNZIP_DIR"

# Make a ephemeral clone of the output RIA
datalad clone \
    -D "Create reckless ephemeral clone of nordic_fmriprep outputs" \
    --reckless ephemeral \
    ria+file://${FMRIPREP_OUTPUT_RIA}#~data \
    ${ZIP_DIR}

cd "$ZIP_DIR" || exit 1

# Unzip each MRIQC zip file into the UNZIP_DIR
for z in *.zip; do
    echo "Unzipping $z ..."
    unzip -o "$z" -d "$UNZIP_DIR"
done
