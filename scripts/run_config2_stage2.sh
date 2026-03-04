#!/usr/bin/env bash
set -euo pipefail

# Before running stage-2, update config/phase2_base_best.py
# to the best architecture from stage-1.
# jobs file currently schedules the 11 LoRA combinations.

python3 scripts/schedule_train.py \
  --jobs-file scripts/config2_stage2_jobs.txt \
  --skip-completed \
  --min-free-mem-mb "${MIN_FREE_MEM_MB:-6000}" \
  --max-gpu-util "${MAX_GPU_UTIL:-25}" \
  --poll-seconds "${POLL_SECONDS:-20}" \
  --max-jobs-per-gpu "${MAX_JOBS_PER_GPU:-1}"
