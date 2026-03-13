#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/wsl_home/works/llms/rnn/recurrent_pythia_mvp
mkdir -p logs
source .venv/bin/activate
EXPERIMENT_LIST="${EXPERIMENT_LIST:-configs/stage1_experiments.txt}"
RUN_NAME="${RUN_NAME:-$(basename "${EXPERIMENT_LIST%.*}")}"
bash ./scripts/run_stage1_all.sh 2>&1 | tee "logs/${RUN_NAME}.log"
