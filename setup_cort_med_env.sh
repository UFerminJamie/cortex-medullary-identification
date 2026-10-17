#!/usr/bin/env bash
#SBATCH --account=sarder-hubmap
#SBATCH --qos=sarder-hubmap
#SBATCH --job-name=cort_med_env
#SBATCH --output=logs/cort_med_env_%j.out
#SBATCH --error=logs/cort_med_env_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load conda

env_dir=${CORT_MED_CONDA_ENV:-/blue/pinaki.sarder/j.fermin/conda/envs/cort_med_gpu}

if [[ ! -f "$env_dir/conda-meta/history" ]]; then
  if [[ -e "$env_dir" ]]; then
    echo "Incomplete or unknown environment exists: $env_dir" >&2
    exit 1
  fi
  conda create -p "$env_dir" -c conda-forge --override-channels python=3.10 pip -y
fi

"$env_dir/bin/python" -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129
"$env_dir/bin/python" -m pip install -r requirements-svc.txt --extra-index-url https://pypi.nvidia.com
"$env_dir/bin/python" -m pip install -e ./TRIDENT
"$env_dir/bin/python" -m pip check
"$env_dir/bin/python" -c "import torch, cupy; from cuml.svm import SVC; assert torch.cuda.is_available(); assert cupy.cuda.runtime.getDeviceCount() == 1; print('Shared TRIDENT/RAPIDS environment ready:', torch.__version__)"
