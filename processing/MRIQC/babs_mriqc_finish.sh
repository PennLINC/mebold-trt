#!/bin/bash

MRIQC_OUTPUT_RIA="/cbica/projects/executive_function/mebold_trt/derivatives/mriqc_babs_project/output_ria"
ZIP_DIR="/cbica/projects/executive_function/mebold_trt/derivatives/mriqc-24-0-2_zipped_ephe"
UNZIP_DIR="/cbica/projects/executive_function/mebold_trt/derivatives/mriqc-24-0-2_unzipped"

mkdir -p "$UNZIP_DIR"

# Make a ephemeral clone of the output RIA
datalad clone \
    -D "Create reckless ephemeral clone of mriqc-24-0-2 outputs" \
    --reckless ephemeral \
    ria+file://${MRIQC_OUTPUT_RIA}#~data \
    ${ZIP_DIR}

cd "$ZIP_DIR" || exit 1

# Unzip each MRIQC zip file into the UNZIP_DIR
for z in *.zip; do
    echo "Unzipping $z ..."
    unzip -o "$z" -d "$UNZIP_DIR"
done
