#!/bin/bash
#SBATCH --job-name=nback_firstlevel
#SBATCH --output=nback_%A_%a.out
#SBATCH --error=nback_%A_%a.err
#SBATCH --array=0-7          # 8 subjects
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# --------------------------
# LOAD ENVIRONMENT
# --------------------------
module purge
eval "$(micromamba shell hook --shell bash)"
micromamba activate nilearn_glm

# --------------------------
# SUBJECT LIST (NO COMMAS)
# --------------------------
SUBJECT_LIST=(01 02 03 04 05 06 07 08)
SUB_ID=${SUBJECT_LIST[$SLURM_ARRAY_TASK_ID]}

echo "Running first-level GLM for subject: $SUB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

# --------------------------
# RUN FIRST-LEVEL SCRIPT
# --------------------------
python /cbica/projects/executive_function/mebold_trt/github/parker/analysis/run_nback_glm_single_subject.py $SUB_ID
