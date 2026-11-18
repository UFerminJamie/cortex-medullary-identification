#!/usr/bin/env bash
#SBATCH --job-name=cort_med_identification
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1

set -euo pipefail
[[ -n ${SLURM_JOB_ID:-} && $# -eq 1 ]] || { echo 'Usage: sbatch run_selected_slides.sh MANIFEST' >&2; exit 2; }
project_dir=${PROJECT_DIR:-$(cd "$(dirname "$0")" && pwd)}
module load conda
conda activate "${CORT_MED_CONDA_ENV:-/blue/pinaki.sarder/j.fermin/conda/envs/cort_med_gpu}"
cd "$project_dir"
python -m scripts.run_selected_slides "$1"
