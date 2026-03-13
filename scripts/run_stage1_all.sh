#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/wsl_home/works/llms/rnn/recurrent_pythia_mvp
mkdir -p logs

EXPERIMENT_LIST="${EXPERIMENT_LIST:-configs/stage1_experiments.txt}"
WANDB_PROJECT="${WANDB_PROJECT:-recurrent-pythia-mvp}"
WANDB_MODE="${WANDB_MODE:-online}"

echo "stage1_experiment_list=${EXPERIMENT_LIST}"
echo "wandb_project=${WANDB_PROJECT}"
echo "wandb_mode=${WANDB_MODE}"

while IFS= read -r config_path; do
  if [[ -z "${config_path}" ]] || [[ "${config_path}" == \#* ]]; then
    continue
  fi

  run_name="$(basename "${config_path%.*}")"
  echo "starting_config=${config_path}"
  echo "run_name=${run_name}"
  bash ./scripts/run_config.sh "${config_path}" "${WANDB_PROJECT}" "${WANDB_MODE}" 2>&1 | tee "logs/${run_name}.log"
  echo "finished_config=${config_path}"
done < "${EXPERIMENT_LIST}"
