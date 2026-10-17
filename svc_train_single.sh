#!/usr/bin/env bash
#SBATCH --account=sarder-hubmap
#SBATCH --qos=sarder-hubmap
#SBATCH --job-name=svc_train_single
#SBATCH --output=logs/svc_train_single_%j.out
#SBATCH --error=logs/svc_train_single_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load conda
conda activate "${CORT_MED_CONDA_ENV:-/blue/pinaki.sarder/j.fermin/conda/envs/cort_med_gpu}"
exec python -m scripts.svc_train_single "$@"
