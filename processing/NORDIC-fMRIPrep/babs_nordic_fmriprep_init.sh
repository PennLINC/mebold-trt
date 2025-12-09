#!/bin/bash

mkdir -p /cbica/projects/executive_function/mebold_trt/templateflow_home
export TEMPLATEFLOW_HOME=/cbica/projects/executive_function/mebold_trt/templateflow_home

babs init \
  --container_ds /cbica/projects/executive_function/mebold_trt/software/apptainer-ds/nordic-fmriprep-ds \
  --container_name nordic-0-0-1 \
  --container_config /cbica/projects/executive_function/mebold_trt/derivatives/code/babs_nordic_fmriprep.yaml \
  --processing_level subject \
  --queue slurm \
  /cbica/projects/executive_function/mebold_trt/derivatives/nordic_fmriprep_babs_project
