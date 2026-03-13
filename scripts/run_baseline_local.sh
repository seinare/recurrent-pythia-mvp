#!/usr/bin/env bash
set -euo pipefail

WANDB_PROJECT="${WANDB_PROJECT:-recurrent-pythia-mvp}"
CONFIG_PATH="${CONFIG_PATH:-configs/baseline_local_128.yaml}"
DEFAULT_RUN_NAME="$(basename "${CONFIG_PATH%.*}")"
WANDB_NAME="${WANDB_NAME:-${DEFAULT_RUN_NAME}}"
WANDB_MODE="${WANDB_MODE:-online}"

python -u -m src.trainer train \
  --config "${CONFIG_PATH}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-name "${WANDB_NAME}" \
  --wandb-mode "${WANDB_MODE}"
