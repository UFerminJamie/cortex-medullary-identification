#!/usr/bin/env bash
#SBATCH --account=sarder-hubmap
#SBATCH --qos=sarder-hubmap
#SBATCH --job-name=generate_ifs
#SBATCH --output=logs/generate_ifs_%j.out
#SBATCH --error=logs/generate_ifs_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load conda
conda activate "${CORT_MED_CONDA_ENV:-/blue/pinaki.sarder/j.fermin/conda/envs/cort_med_gpu}"

exec python utils/generate_IFS.py "$@"

# Generate both IFS annotations:
# sbatch generate_IFS.sh /orange/pinaki.sarder/j.fermin/cort_med_wsi/trident_processed_final/KPMP/contours_geojson /path/to/ftu_directory /path/to/output_dir
# Generate only the mask or contours by adding --mode mask or --mode contours.
# Add --overwrite to replace existing output files.
