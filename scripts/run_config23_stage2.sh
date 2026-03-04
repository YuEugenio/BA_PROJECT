#!/usr/bin/env bash
set -euo pipefail

# Stage-2 LoRA sweep on config23 architecture.
# This script runs in foreground and auto-dispatches jobs to available GPUs.

python3 scripts/schedule_train.py \
  --jobs-file scripts/stage2_config23_jobs.txt \
  --skip-completed \
  --min-free-mem-mb "${MIN_FREE_MEM_MB:-6000}" \
  --max-gpu-util "${MAX_GPU_UTIL:-25}" \
  --poll-seconds "${POLL_SECONDS:-20}" \
  --max-jobs-per-gpu "${MAX_JOBS_PER_GPU:-1}"
