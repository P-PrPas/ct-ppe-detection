#!/usr/bin/env bash
set -euo pipefail
cd /root/P/PPE-Detection
export LD_LIBRARY_PATH=/root/P/PPE-Detection/.runtime/nvidia-580.95.05/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=disabled
exec .venv-ppe-v100/bin/python -u scripts/train_ppe_four_more.py >> reports/ppe-expansion-build-20261116/train.log 2>&1
