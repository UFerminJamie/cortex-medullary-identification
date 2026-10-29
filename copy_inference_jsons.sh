#!/usr/bin/env bash
#SBATCH --account=sarder-hubmap
#SBATCH --qos=sarder-hubmap
#SBATCH --job-name=copy_inference_jsons
#SBATCH --output=logs/copy_inference_jsons_%j.out
#SBATCH --error=logs/copy_inference_jsons_%j.err
#SBATCH --time=0:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load conda
conda activate "${CORT_MED_CONDA_ENV:-/blue/pinaki.sarder/j.fermin/conda/envs/cort_med_gpu}"

exec python utils/copy_inference_jsons.py "$@"

# Run with default suffixes:
# sbatch copy_inference_jsons.sh /path/to/input_dir /path/to/out_dir
# Specify endings if needed:
# sbatch copy_inference_jsons.sh /path/to/input_dir /path/to/out_dir --suffixes _refined_heatmap.json _mask_gt.json
