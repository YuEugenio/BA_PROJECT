#!/usr/bin/env bash
set -euo pipefail

# Auto-schedule CLIPViT 3-stream concat linear + attn_output LoRA tuning experiments.
# Foreground execution, GPU-aware dispatch.
# Optional:
#   JOBS_FILE=scripts/clipvit_attnproj_tuning_lora_jobs.txt bash scripts/run_clipvit_attnproj_tuning.sh

python3 scripts/schedule_train.py \
  --jobs-file "${JOBS_FILE:-scripts/clipvit_attnproj_tuning_all_jobs.txt}" \
  --skip-completed \
  --min-free-mem-mb "${MIN_FREE_MEM_MB:-6000}" \
  --max-gpu-util "${MAX_GPU_UTIL:-25}" \
  --poll-seconds "${POLL_SECONDS:-20}" \
  --max-jobs-per-gpu "${MAX_JOBS_PER_GPU:-1}"
