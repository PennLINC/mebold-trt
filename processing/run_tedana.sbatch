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

# Override the array range at submit time to match the number of subject-session pairs:
#   sbatch --array=1-$(find /cbica/projects/executive_function/mebold_trt/ds005250/sub-* -mindepth 1 -maxdepth 1 -type d -name 'ses-*' | wc -l) processing/run_tedana.sbatch

eval "$(micromamba shell hook --shell bash)"
micromamba activate meboldtrt

set -euo pipefail
pwd; hostname; date

RAW_DIR="/cbica/projects/executive_function/mebold_trt/ds005250"
FMRIPREP_DIR="/cbica/projects/executive_function/mebold_trt/derivatives/nordic_fmriprep_unzipped/fmriprep"
TEMP_DIR="/cbica/comp_space/executive_function/tedana_temp"
TEDANA_OUT_DIR="/cbica/projects/executive_function/mebold_trt/derivatives/tedana"
CODE_DIR="/cbica/projects/executive_function/mebold_trt/code"

mkdir -p "${TEMP_DIR}"
mkdir -p "${CODE_DIR}/processing/jobs"

# Build subject-session pairs (e.g., sub-01_ses-1)
mapfile -t subj_sess_pairs < <(
  find "${RAW_DIR}"/sub-* -mindepth 1 -maxdepth 1 -type d -name "ses-*" \
    | awk -F/ '{n=split($0,a,\"/\"); sub=a[n-1]; ses=a[n]; print sub\"_\"ses}' \
    | sort
)
pair_count=${#subj_sess_pairs[@]}

if [[ ${pair_count} -eq 0 ]]; then
  echo "No subject-session pairs found under ${RAW_DIR}"
  exit 1
fi

task_index=$((SLURM_ARRAY_TASK_ID - 1))
if [[ ${task_index} -lt 0 || ${task_index} -ge ${pair_count} ]]; then
  echo "SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID} outside pair list (n=${pair_count}); exiting."
  exit 0
fi

pair=${subj_sess_pairs[${task_index}]}
subject_label=${pair%%_*}
session_label=${pair##*_}
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

