#!/usr/bin/env bash
set -euo pipefail

robotwin_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export REPO_PATH=${REPO_PATH:-/workspace/RLinf}
export EMBODIED_PATH="$REPO_PATH/examples/embodiment"
export ROBOTWIN_PATH="$robotwin_root"
export ASSETS_PATH="$robotwin_root"
export ROBOTWIN_ASSETS_PATH=${ROBOTWIN_ASSETS_PATH:-/data/robotwin_assets}
export SAPIEN_RUNTIME=${SAPIEN_RUNTIME:-/workspace/ctr_backup/sapien300b1}

if [[ ! ${GPU_LIST:-} =~ ^[0-9]+$ ]]; then
  echo 'Set GPU_LIST to the one GPU you have reserved before running.' >&2
  exit 2
fi
if [[ ! -d "$SAPIEN_RUNTIME/sapien" ]]; then
  echo "Missing reference SAPIEN runtime: $SAPIEN_RUNTIME" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export RAY_ADDRESS=${RAY_ADDRESS:-local}
export RAY_OVERRIDE_RESOURCES=${RAY_OVERRIDE_RESOURCES:-'{"CPU":8,"object_store_memory":1073741824}'}
export ROBOTWIN_RENDER_DEVICE=${ROBOTWIN_RENDER_DEVICE:-cuda:0}
export PYTHONPATH="$SAPIEN_RUNTIME:$robotwin_root:$REPO_PATH${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/data/robotwin_ckpt/openpi-cache}
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

cd "$robotwin_root"
exec /opt/venv/openpi/bin/python -m rlinf_eval.cli "$@"
