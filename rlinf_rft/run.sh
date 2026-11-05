#!/usr/bin/env bash
set -euo pipefail

robotwin_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export REPO_PATH=${REPO_PATH:-/workspace/RLinf}
export EMBODIED_PATH="$REPO_PATH/examples/embodiment"
export ROBOTWIN_PATH="$robotwin_root"
export ROBOT_PLATFORM=ALOHA
export ASSETS_PATH="$robotwin_root"
export ROBOTWIN_ASSETS_PATH=${ROBOTWIN_ASSETS_PATH:-/data/robotwin_assets}
export SAPIEN_RUNTIME=${SAPIEN_RUNTIME:-/workspace/ctr_backup/sapien300b1}
export PYTHONPATH="$SAPIEN_RUNTIME:$robotwin_root:$REPO_PATH${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/data/robotwin_ckpt/openpi-cache}
export RAY_ADDRESS=local
export RAY_OVERRIDE_RESOURCES=${RAY_OVERRIDE_RESOURCES:-'{"CPU":32,"object_store_memory":2147483648}'}
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export ROBOTWIN_RENDER_DEVICE=${ROBOTWIN_RENDER_DEVICE:-cuda:0}
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

case "${1:-}" in
  scan|train|eval|smoke)
    : "${GPU_LIST:?Set GPU_LIST to the confirmed physical GPU indices}"
    export CUDA_VISIBLE_DEVICES="$GPU_LIST"
    if [[ ! -d "$SAPIEN_RUNTIME/sapien" ]]; then
      echo "Missing SAPIEN reference runtime: $SAPIEN_RUNTIME" >&2
      exit 2
    fi
    ;;
  *) export CUDA_VISIBLE_DEVICES= ;;
esac

cd "$robotwin_root"
if [[ "${1:-}" == workflow ]]; then
  shift
  exec /opt/venv/openpi/bin/python -B -m rlinf_rft.workflow "$@"
fi
if [[ "${1:-}" == smoke ]]; then
  shift
  exec /opt/venv/openpi/bin/python -B -m rlinf_rft.smoke "$@"
fi
exec /opt/venv/openpi/bin/python -B -m rlinf_rft.cli "$@"
