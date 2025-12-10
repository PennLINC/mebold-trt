#!/bin/bash

XCPD_SE_OUTPUT_RIA="/cbica/projects/executive_function/mebold_trt/derivatives/xcpd_SE_babs_project/output_ria"
ZIP_DIR="/cbica/projects/executive_function/mebold_trt/derivatives/xcpd_SE_zipped_ephe"
UNZIP_DIR="/cbica/projects/executive_function/mebold_trt/derivatives/xcpd_SE_unzipped"

mkdir -p "$UNZIP_DIR"

# Make a ephemeral clone of the output RIA
datalad clone \
    -D "Create reckless ephemeral clone of xcpd_SE outputs" \
    --reckless ephemeral \
    ria+file://${XCPD_SE_OUTPUT_RIA}#~data \
    ${ZIP_DIR}

cd "$ZIP_DIR" || exit 1

# Unzip each MRIQC zip file into the UNZIP_DIR
for z in *.zip; do
    echo "Unzipping $z ..."
    unzip -o "$z" -d "$UNZIP_DIR"
done
