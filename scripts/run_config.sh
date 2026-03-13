#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/wsl_home/works/llms/rnn/recurrent_pythia_mvp
source .venv/bin/activate

CONFIG_PATH="${1:?usage: run_config.sh <config-path> [wandb-project] [wandb-mode]}"
WANDB_PROJECT="${2:-recurrent-pythia-mvp}"
WANDB_MODE="${3:-online}"
RUN_NAME="${RUN_NAME:-$(basename "${CONFIG_PATH%.*}")}"

python -u -m src.trainer train \
  --config "${CONFIG_PATH}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-name "${RUN_NAME}" \
  --wandb-mode "${WANDB_MODE}"
