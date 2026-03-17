# Recurrent Pythia MVP Status

Last updated: 2026-03-17

This file mirrors the current project state so the repository remains self-contained on GitHub. The original planning document lives at `../agent/AGENT.md`.

## Implemented

- Byte-level `enwik8` dataset with small-corpus split fallback
- Full attention, local attention, and recurrent local attention training paths
- RoPE-based position encoding for attention
- Layer-local GRU recurrence with configurable `gru_hidden`
- GRU output projection back into transformer hidden space
- Default GRU output `RMSNorm`
- Configurable GRU placement: `pre_attn` or `post_attn`
- Optional first-layer GRU disable
- Optional prefix chunk prepended to each sample
- Config-driven training, W&B logging, tmux batch launchers, and resolved-config snapshots

## Current Naming Convention

- Example recurrent run: `rnn_w128_l1024_gru512_skipfirst_prefix`
- Meaning:
  - `w128`: local attention window 128
  - `l1024`: training length 1024
  - `gru512`: recurrent state size 512
  - `skipfirst`: first layer GRU disabled
  - `prefix`: prepend one window-length prefix chunk

## Current Findings

- Historical `baseline_full` outperformed earlier local and recurrent baselines
- Early recurrent variants only slightly improved over local-only
- Full BPTT without additional structural changes did not help by itself
- Adding GRU output norm, prefix chunking, and revised GRU placement improved recurrent results significantly
- The strongest recent recurrent variants reached roughly `eval/bpc ~= 1.44`

## Active Experiment Families

- `baseline_full`
- `baseline_local_128`
- `rnn_w128_l1024_gru16_skipfirst_prefix`
- `rnn_w128_l1024_gru32_skipfirst_prefix`
- `rnn_w128_l1024_gru64_skipfirst_prefix`
- `rnn_w128_l1024_gru128_skipfirst_prefix`
- `rnn_w128_l1024_gru256_skipfirst_prefix`
- `rnn_w128_l1024_gru512_skipfirst_prefix`

## Open Questions

- Best recurrent fusion strategy under the current normalized setup
- Best `gru_hidden` size for the skipfirst-prefix configuration
- Fair comparison against baselines rerun under the current code rather than older logs
