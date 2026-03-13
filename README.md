# Layer-Local Recurrent Pythia-70M MVP

This repository bootstraps the MVP described in [agent/AGENT.md](/D:/wsl_home/works/llms/rnn/agent/AGENT.md).

Current status:

- Project scaffold created
- `src/dataset.py` implemented first, as requested by the spec
- A minimal trainable transformer stack is wired up for full attention, local attention, and local attention plus GRU recurrence

## Layout

```text
recurrent_pythia_mvp/
|-- configs/
|-- data/
|-- src/
|   |-- __init__.py
|   |-- dataset.py
|   |-- modeling_recurrent.py
|   |-- trainer.py
|   `-- eval_tasks.py
|-- scripts/
|   |-- run_config.sh
|   |-- run_baseline_full.sh
|   |-- run_baseline_local.sh
|   |-- run_recurrent.sh
|   |-- run_stage1_all.sh
|   `-- run_stage1_all_tmux.sh
`-- requirements.txt
```

## Quick start

From WSL:

```bash
cd /mnt/d/wsl_home/works/llms/rnn/recurrent_pythia_mvp
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
python -m src.trainer list-configs
python -m src.trainer train --config configs/recurrent_local_128_gru.yaml
bash scripts/run_stage1_all.sh
```

If `enwik8` is not present yet, use `data/download_enwik8.sh`.

## Notes

- The dataset operates on raw corpus bytes. On `enwik8`, the vocabulary is the set of observed bytes in the corpus.
- `--window-size 0` means full causal attention. Positive values enable local causal attention with that window.
- `--no-recurrence` disables the per-layer GRU path. Without that flag, training uses chunk streaming with detached recurrent state across chunks.
- `train --config ...` loads YAML/JSON experiment files, and CLI flags still override config values.
- Each run writes a resolved parameter snapshot to `logs/<run_name>.resolved.yaml`, while the tmux launcher writes stdout to `logs/<run_name>.log`.
- `configs/stage1_experiments.txt` is the current batch list for this phase.
