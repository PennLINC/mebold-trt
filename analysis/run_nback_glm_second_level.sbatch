#!/bin/bash
#SBATCH --job-name=nback_secondlevel
#SBATCH --output=nback_secondlevel_%j.out
#SBATCH --error=nback_secondlevel_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

eval "$(micromamba shell hook --shell bash)"
micromamba activate nilearn_glm

python /cbica/projects/executive_function/mebold-trt/code/analysis/run_nback_second_level.py
