#!/usr/bin/env bash
#SBATCH --account=sarder-hubmap
#SBATCH --qos=sarder-hubmap
#SBATCH --job-name=trident_batch
#SBATCH --output=logs/trident_batch_%j.out
#SBATCH --error=logs/trident_batch_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: sbatch run_batch_slide.sh WSI_DIR JOB_DIR [TRIDENT_FLAGS...]

Requests one L4 GPU on HiPerGator, then runs TRIDENT segmentation,
tissue tile generation, and patch feature extraction. Nested WSI folders
are searched by default.
Defaults are HEST, UNI-v1, 40x, 512 px, 384 px overlap, and GPU 0.
Submit from the project directory. Set TRIDENT_DIR to use another checkout,
or TRIDENT_CONDA_ENV to select the Conda environment.
Additional flags are passed to run_batch_of_slides.py and can override defaults.

Example:
  sbatch run_batch_slide.sh ./wsi_dir ./output_dir
EOF
}

if [[ $# -lt 2 || ${1:-} == --help || ${1:-} == -h ]]; then
  usage
  [[ ${1:-} == --help || ${1:-} == -h ]] && exit 0
  exit 2
fi

if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "Submit this script with sbatch to request HiPerGator resources." >&2
  exit 2
fi

wsi_dir=$1
job_dir=$2
shift 2

project_dir=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
trident_dir=${TRIDENT_DIR:-"$project_dir/TRIDENT"}
trident_python=${TRIDENT_PYTHON:-python}
trident_env=${TRIDENT_CONDA_ENV:-${CORT_MED_CONDA_ENV:-/blue/pinaki.sarder/j.fermin/conda/envs/cort_med_gpu}}
read -r -a gpu_ids <<< "${GPUS:-0}"
if [[ ${#gpu_ids[@]} -eq 0 ]]; then
  echo "GPUS must list at least one allocated GPU index." >&2
  exit 2
fi

if [[ ! -f "$trident_dir/run_batch_of_slides.py" ]]; then
  echo "TRIDENT runner not found: $trident_dir/run_batch_of_slides.py" >&2
  exit 2
fi
if [[ ! -d "$wsi_dir" ]]; then
  echo "WSI directory not found: $wsi_dir" >&2
  exit 2
fi

module load conda
conda activate "$trident_env"

python3 "$trident_dir/run_batch_of_slides.py" \
  --task all \
  --wsi_dir "$wsi_dir" \
  --job_dir "$job_dir" \
  --search_nested \
  --segmenter "${SEGMENTER:-hest}" \
  --patch_encoder "${PATCH_ENCODER:-uni_v1}" \
  --mag "${MAG:-40}" \
  --patch_size "${PATCH_SIZE:-512}" \
  --overlap "${OVERLAP:-384}" \
  --gpus "${gpu_ids[@]}" \
  --min_tissue_proportion "${MIN_TISSUE_PROPORTION:-0.5}" \
  --max_workers "${MAX_WORKERS:-2}" \
  --clear_dead_locks \
  --wsi_ext "${WSI_EXT:-.svs}" \
  "$@"

python3 "$project_dir/scripts/normalize_contour_names.py" \
  "$job_dir/contours_geojson"
