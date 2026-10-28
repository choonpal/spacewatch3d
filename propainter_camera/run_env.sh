#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_MODULE_LOADING=LAZY MPLBACKEND=Agg PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/workspace/v1_splatfacto_comparison_20261025/system-libs/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=/workspace/panogrf_propainter_20261027/extra_packages${PYTHONPATH:+:$PYTHONPATH}
exec /workspace/grounded_sam2_panorama_test_20261023/env/bin/python "$@"
