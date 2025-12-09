#!/bin/bash
babs init \
  --container_ds /cbica/projects/executive_function/mebold_trt/software/apptainer-ds/mriqc-24-0-2-ds \
  --container_name mriqc-24-0-2 \
  --container_config /cbica/projects/executive_function/mebold_trt/derivatives/code/babs_mriqc.yaml \
  --processing_level subject \
  --queue slurm \
  /cbica/projects/executive_function/mebold_trt/derivatives/mriqc_babs_project
