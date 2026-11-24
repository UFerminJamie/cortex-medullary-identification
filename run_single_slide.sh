#!/usr/bin/env bash
#SBATCH --job-name=cort_med_single
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1

set -euo pipefail

if [[ ${CORT_MED_CONTAINER:-0} != 1 && -z ${SLURM_JOB_ID:-} ]]; then
  echo 'Submit through sbatch; an L4 allocation is required.' >&2
  exit 2
fi
if [[ $# -ne 2 ]]; then
  echo 'Usage: run_single_slide.sh SLIDE_PATH JOB_DIR' >&2
  exit 2
fi

slide_path=$1
job_dir=$2
slide_ext=.${slide_path##*.}
shopt -s nullglob nocaseglob
same_type=("$(dirname "$slide_path")"/*"$slide_ext")
shopt -u nullglob nocaseglob
[[ ${#same_type[@]} -eq 1 ]] || { echo 'The single-slide input directory must contain exactly one slide of this type.' >&2; exit 2; }
project_dir=${PROJECT_DIR:-$(cd "$(dirname "$0")" && pwd)}
trident_dir=${TRIDENT_DIR:-$project_dir/TRIDENT}
trident_env=${TRIDENT_CONDA_ENV:-${CORT_MED_CONDA_ENV:-/blue/pinaki.sarder/j.fermin/conda/envs/cort_med_gpu}}
runner=$trident_dir/run_batch_of_slides.py
[[ -f $slide_path && -f $runner ]] || { echo 'Slide or TRIDENT batch runner is missing.' >&2; exit 2; }

if [[ ${CORT_MED_CONTAINER:-0} != 1 ]]; then
  module load conda
  conda activate "$trident_env"
  python_bin=$(command -v python)
else
  python_bin=${CORT_MED_CONDA_ENV:-/opt/conda/envs/cort_med_gpu}/bin/python
  [[ -x $python_bin ]] || { echo "Container Conda Python not found: $python_bin" >&2; exit 2; }
fi
help_text=$("$python_bin" "$runner" --help)
for flag in --task --wsi_dir --job_dir --mag --patch_size --overlap --patch_encoder --segmenter --gpus; do
  [[ $help_text == *"$flag"* ]] || { echo "Incompatible TRIDENT runner: missing $flag" >&2; exit 2; }
done
if [[ -d $trident_dir/.git ]]; then
  echo "TRIDENT commit: $(git -C "$trident_dir" rev-parse HEAD)"
fi
"$python_bin" "$runner" --task all --wsi_dir "$(dirname "$slide_path")" --job_dir "$job_dir" \
  --wsi_ext "$slide_ext" \
  --segmenter "${SEGMENTER:-hest}" --patch_encoder "${PATCH_ENCODER:-uni_v1}" \
  --mag "${CORT_MED_MAGNIFICATION:-40}" \
  --patch_size "${CORT_MED_PATCH_SIZE:-512}" \
  --overlap "${CORT_MED_TILE_OVERLAP:-384}" --gpus 0 \
  --min_tissue_proportion "${MIN_TISSUE_PROPORTION:-0.5}" \
  --max_workers "${MAX_WORKERS:-2}"
"$python_bin" "$project_dir/scripts/normalize_contour_names.py" "$job_dir/contours_geojson"
stem=$(basename "$slide_path")
stem=${stem%.*}
mag=${CORT_MED_MAGNIFICATION:-40}
mag=${mag%.*}
feature=$job_dir/${mag}x_${CORT_MED_PATCH_SIZE:-512}px_${CORT_MED_TILE_OVERLAP:-384}px_overlap/features_${PATCH_ENCODER:-uni_v1}/$stem.h5
[[ -s $feature ]] || { echo "Expected feature file missing: $feature" >&2; exit 1; }
"$python_bin" - "$feature" "${CORT_MED_TILE_OVERLAP:-384}" "${CORT_MED_PATCH_SIZE:-512}" <<'PY'
import h5py
import sys

with h5py.File(sys.argv[1], "r") as handle:
    attrs = handle["coords"].attrs
    expected_overlap = int(sys.argv[2])
    expected_patch_size = int(sys.argv[3])
    if int(attrs.get("overlap", -1)) != expected_overlap or int(attrs.get("patch_size", -1)) != expected_patch_size:
        raise SystemExit("TRIDENT feature geometry does not match configured patch size / overlap")
PY
echo "PREPROCESSED $stem $feature"
