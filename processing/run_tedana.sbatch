#!/bin/bash
#SBATCH --job-name=tedana
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-17
#SBATCH --cpus-per-task=8
#SBATCH --mem=24gb
#SBATCH --time=24:00:00
# Outputs ----------------------------------
#SBATCH --output=/cbica/projects/executive_function/mebold_trt/code/processing/jobs/tedana_%A-%a.out
#SBATCH --error=/cbica/projects/executive_function/mebold_trt/code/processing/jobs/tedana_%A-%a.err
# ------------------------------------------

eval "$(micromamba shell hook --shell bash)"
micromamba activate meboldtrt

set -euo pipefail
pwd; hostname; date

RAW_DIR="/cbica/projects/executive_function/mebold_trt/ds005250"
FMRIPREP_DIR="/cbica/projects/executive_function/mebold_trt/derivatives/nordic_fmriprep_unzipped/fmriprep"
TEMP_DIR="/cbica/comp_space/executive_function/tedana_temp"
TEDANA_OUT_DIR="/cbica/projects/executive_function/mebold_trt/derivatives/tedana"
CODE_DIR="/cbica/projects/executive_function/mebold_trt/code"
PAIRS_TSV="${CODE_DIR}/processing/jobs/tedana_pairs.tsv"

mkdir -p "${TEMP_DIR}"
mkdir -p "${CODE_DIR}/processing/jobs"

mapfile -t subject_list < <(tail -n +2 "${PAIRS_TSV}" | cut -f1)
mapfile -t session_list < <(tail -n +2 "${PAIRS_TSV}" | cut -f2)
pair_count=${#subject_list[@]}

task_index=$((SLURM_ARRAY_TASK_ID - 1))
if [[ ${task_index} -lt 0 || ${task_index} -ge ${pair_count} ]]; then
  echo "SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID} outside pair list (n=${pair_count}); exiting."
  exit 1
fi

subject_label=${subject_list[${task_index}]}
session_label=${session_list[${task_index}]}
echo "Processing ${subject_label} ${session_label} (task ${SLURM_ARRAY_TASK_ID}/${pair_count})"

cmd="python ${CODE_DIR}/processing/run_tedana.py \
    --raw-dir ${RAW_DIR} \
    --fmriprep-dir ${FMRIPREP_DIR} \
    --temp-dir ${TEMP_DIR} \
    --tedana-out-dir ${TEDANA_OUT_DIR} \
    --session-label ${session_label} \
    --subject-label ${subject_label}"

echo "Commandline: ${cmd}"
eval "${cmd}"
exitcode=$?

echo "session-${session_label}   ${SLURM_ARRAY_TASK_ID}    ${exitcode}" \
      >> ${CODE_DIR}/processing/jobs/"${SLURM_JOB_NAME}"."${SLURM_ARRAY_JOB_ID}".tsv
echo "Finished task ${SLURM_ARRAY_TASK_ID} with exit code ${exitcode}"
exit ${exitcode}

