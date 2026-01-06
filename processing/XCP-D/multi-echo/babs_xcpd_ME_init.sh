#!/bin/bash
babs init \
  --container_ds /cbica/projects/executive_function/mebold_trt/software/apptainer-ds/xcpd-0-13-0-ds \
  --container_name xcpd-0-13-0 \
  --container_config /cbica/projects/executive_function/mebold_trt/derivatives/code/babs_xcpd_ME.yaml \
  --processing_level subject \
  --queue slurm \
  /cbica/projects/executive_function/mebold_trt/derivatives/xcpd_ME_babs_project
