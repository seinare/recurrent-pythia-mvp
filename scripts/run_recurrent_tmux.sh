#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/wsl_home/works/llms/rnn/recurrent_pythia_mvp
mkdir -p logs
source .venv/bin/activate
CONFIG_PATH="${CONFIG_PATH:-configs/recurrent_local_128_gru.yaml}"
RUN_NAME="${RUN_NAME:-$(basename "${CONFIG_PATH%.*}")}"
bash ./scripts/run_recurrent.sh 2>&1 | tee "logs/${RUN_NAME}.log"
