#!/usr/bin/env bash
set -euo pipefail

# Usage example:
#   bash scripts/run_config2_stage1.sh
# Optional env overrides:
#   MIN_FREE_MEM_MB=14000 MAX_GPU_UTIL=20 POLL_SECONDS=20 MAX_JOBS_PER_GPU=1

python3 scripts/schedule_train.py \
  --jobs-file scripts/config2_stage1_jobs.txt \
  --skip-completed \
  --min-free-mem-mb "${MIN_FREE_MEM_MB:-6000}" \
  --max-gpu-util "${MAX_GPU_UTIL:-25}" \
  --poll-seconds "${POLL_SECONDS:-20}" \
  --max-jobs-per-gpu "${MAX_JOBS_PER_GPU:-1}"
