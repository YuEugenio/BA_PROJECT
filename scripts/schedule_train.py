#!/usr/bin/env python3
"""
GPU-aware experiment scheduler for `train.py`.

Features:
- Auto queue from explicit config names, jobs file, and module globs.
- Resource-aware dispatch with `nvidia-smi` (memory/utilization thresholds).
- Fills all currently available GPU slots; waits when no slot is available.
- Per-job log files and JSON state snapshots.
- Optional skip for already completed runs.
- Optional retry on failure.

Example:
  python scripts/schedule_train.py --module-glob "s1_*"
  python scripts/schedule_train.py --module-glob "p2_*"
  python scripts/schedule_train.py --configs config1 config2 config3
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Job:
    config: str
    attempt: int = 1


@dataclass
class RunningJob:
    job: Job
    gpu_id: Optional[int]
    process: subprocess.Popen
    log_file_handle: object
    log_path: Path
    start_time: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-schedule training jobs across available GPUs.")
    parser.add_argument("--configs", nargs="*", default=[], help="Explicit config names.")
    parser.add_argument(
        "--jobs-file",
        type=str,
        default=None,
        help="Path to a text file with one config name per line.",
    )
    parser.add_argument(
        "--module-glob",
        action="append",
        default=[],
        help="Glob over config module names discovered under config/. Repeatable.",
    )
    parser.add_argument(
        "--config-root",
        type=str,
        default="config",
        help="Config root folder to scan for module names.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to run train.py.",
    )
    parser.add_argument(
        "--train-script",
        type=str,
        default="train.py",
        help="Training entry script path.",
    )
    parser.add_argument(
        "--extra-args",
        type=str,
        default="",
        help="Extra args appended to each train command, e.g. \"--batch_size 4\".",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Value passed to `--device` in train.py (default: cuda).",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip configs that already have at least one completed run in outputs/<config>/.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Max retry times after first failure. 0 means no retry.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=20,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--min-free-mem-mb",
        type=int,
        default=6000,
        help="GPU slot is considered free only if free memory >= this threshold.",
    )
    parser.add_argument(
        "--max-gpu-util",
        type=int,
        default=25,
        help="GPU slot is considered free only if utilization <= this threshold.",
    )
    parser.add_argument(
        "--max-jobs-per-gpu",
        type=int,
        default=1,
        help="Max scheduler-launched jobs per GPU.",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="If no GPU is available or nvidia-smi is absent, run one job at a time on CPU/device arg.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only print resolved queue and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show scheduling decisions without actually launching train.py.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="outputs/scheduler_logs",
        help="Directory to store scheduler logs and state files.",
    )
    return parser.parse_args()


def discover_module_names(config_root: Path) -> List[str]:
    modules: List[str] = []
    for py_file in sorted(config_root.rglob("*.py")):
        if py_file.name == "__init__.py":
            continue
        rel = py_file.relative_to(config_root)
        mod = str(rel.with_suffix("")).replace(os.sep, ".")
        modules.append(mod)
    return modules


def read_jobs_file(path: Path) -> List[str]:
    jobs: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        jobs.append(s)
    return jobs


def resolve_configs(args: argparse.Namespace, repo_root: Path) -> List[str]:
    config_root = repo_root / args.config_root
    discovered = discover_module_names(config_root)

    queue: List[str] = []
    queue.extend(args.configs)

    if args.jobs_file:
        queue.extend(read_jobs_file(repo_root / args.jobs_file))

    if args.module_glob:
        for pattern in args.module_glob:
            matched = [m for m in discovered if fnmatch.fnmatch(m, pattern)]
            queue.extend(sorted(matched))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in queue:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def has_completed_output(repo_root: Path, config_name: str) -> bool:
    cfg_dir = repo_root / "outputs" / config_name
    if not cfg_dir.exists():
        return False
    for run_dir in cfg_dir.glob("run_*"):
        if (run_dir / "best_results.json").exists() and (run_dir / "training_history.json").exists():
            return True
    return False


def query_gpus() -> List[Dict[str, int]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        idx, total, used, util = map(int, parts)
        gpus.append(
            {
                "index": idx,
                "memory_total_mb": total,
                "memory_used_mb": used,
                "memory_free_mb": max(total - used, 0),
                "utilization": util,
            }
        )
    return gpus


def available_gpu_slots(
    gpu_stats: List[Dict[str, int]],
    running: Dict[int, List[RunningJob]],
    min_free_mem_mb: int,
    max_gpu_util: int,
    max_jobs_per_gpu: int,
) -> List[int]:
    slots: List[int] = []
    for g in gpu_stats:
        gpu_id = g["index"]
        local_running = len(running.get(gpu_id, []))
        if local_running >= max_jobs_per_gpu:
            continue
        if g["memory_free_mb"] < min_free_mem_mb:
            continue
        if g["utilization"] > max_gpu_util:
            continue
        free_slots = max_jobs_per_gpu - local_running
        slots.extend([gpu_id] * free_slots)
    return slots


def start_job(
    repo_root: Path,
    args: argparse.Namespace,
    run_log_dir: Path,
    job: Job,
    gpu_id: Optional[int],
    job_index: int,
) -> RunningJob:
    safe_name = job.config.replace("/", "_").replace(".", "_")
    log_path = run_log_dir / f"{job_index:03d}_{safe_name}_try{job.attempt}.log"
    cmd = [args.python, args.train_script, "--config", job.config, "--device", args.device]
    if args.extra_args:
        cmd.extend(shlex.split(args.extra_args))

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if args.dry_run:
        class DummyProc:
            pid = -1
            returncode = 0

            @staticmethod
            def poll():
                return 0

            @staticmethod
            def wait(timeout=None):
                return 0

            @staticmethod
            def terminate():
                return None

            @staticmethod
            def kill():
                return None

        log_fh = open(log_path, "w", encoding="utf-8")
        log_fh.write("DRY RUN\n")
        log_fh.write("CMD: " + " ".join(shlex.quote(x) for x in cmd) + "\n")
        log_fh.flush()
        proc = DummyProc()
    else:
        log_fh = open(log_path, "w", encoding="utf-8")
        log_fh.write("CMD: " + " ".join(shlex.quote(x) for x in cmd) + "\n")
        log_fh.write(f"GPU_SLOT: {gpu_id}\n")
        log_fh.write(f"START_AT: {datetime.now().isoformat()}\n\n")
        log_fh.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )

    return RunningJob(
        job=job,
        gpu_id=gpu_id,
        process=proc,
        log_file_handle=log_fh,
        log_path=log_path,
        start_time=time.time(),
    )


def write_state(
    state_path: Path,
    pending: List[Job],
    running: Dict[int, List[RunningJob]],
    completed: List[Dict[str, object]],
    failed: List[Dict[str, object]],
):
    serial_running = []
    for gpu_id, jobs in running.items():
        for r in jobs:
            serial_running.append(
                {
                    "config": r.job.config,
                    "attempt": r.job.attempt,
                    "gpu_id": gpu_id,
                    "pid": getattr(r.process, "pid", None),
                    "log_path": str(r.log_path),
                    "elapsed_sec": round(time.time() - r.start_time, 1),
                }
            )
    payload = {
        "timestamp": datetime.now().isoformat(),
        "pending": [{"config": j.config, "attempt": j.attempt} for j in pending],
        "running": serial_running,
        "completed": completed,
        "failed": failed,
    }
    state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    queue = resolve_configs(args, repo_root)
    if not queue:
        print("No configs resolved. Use --configs / --jobs-file / --module-glob.")
        return 1

    if args.skip_completed:
        queue = [c for c in queue if not has_completed_output(repo_root, c)]

    if not queue:
        print("All resolved configs are already completed (or queue empty after filtering).")
        return 0

    if args.list_only:
        print("Resolved queue:")
        for i, c in enumerate(queue, start=1):
            print(f"{i:03d}. {c}")
        return 0

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = repo_root / args.log_dir / f"run_{run_stamp}"
    run_log_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_log_dir / "scheduler_state.json"

    pending: List[Job] = [Job(config=c, attempt=1) for c in queue]
    running: Dict[int, List[RunningJob]] = {}
    cpu_running: List[RunningJob] = []
    completed: List[Dict[str, object]] = []
    failed: List[Dict[str, object]] = []
    job_counter = 1
    stop_requested = False

    def handle_signal(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"\\nReceived signal {signum}, will stop after current loop and terminate running jobs.")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"Scheduler started. Total pending jobs: {len(pending)}")
    print(f"Logs: {run_log_dir}")

    while pending or any(running.values()) or cpu_running:
        # Reap GPU jobs
        for gpu_id in list(running.keys()):
            keep: List[RunningJob] = []
            for r in running[gpu_id]:
                rc = r.process.poll()
                if rc is None:
                    keep.append(r)
                    continue

                elapsed = round(time.time() - r.start_time, 1)
                r.log_file_handle.write(f"\\nEND_AT: {datetime.now().isoformat()}\\n")
                r.log_file_handle.write(f"RETURN_CODE: {rc}\\n")
                r.log_file_handle.flush()
                r.log_file_handle.close()

                if rc == 0:
                    completed.append(
                        {
                            "config": r.job.config,
                            "attempt": r.job.attempt,
                            "gpu_id": gpu_id,
                            "elapsed_sec": elapsed,
                            "log_path": str(r.log_path),
                        }
                    )
                    print(f"[DONE] {r.job.config} on GPU {gpu_id} ({elapsed}s)")
                else:
                    if r.job.attempt <= args.max_retries:
                        next_try = Job(config=r.job.config, attempt=r.job.attempt + 1)
                        pending.append(next_try)
                        print(
                            f"[RETRY] {r.job.config} failed on GPU {gpu_id} (rc={rc}), "
                            f"queued retry {next_try.attempt}"
                        )
                    else:
                        failed.append(
                            {
                                "config": r.job.config,
                                "attempt": r.job.attempt,
                                "gpu_id": gpu_id,
                                "return_code": rc,
                                "elapsed_sec": elapsed,
                                "log_path": str(r.log_path),
                            }
                        )
                        print(f"[FAIL] {r.job.config} on GPU {gpu_id} (rc={rc})")
            if keep:
                running[gpu_id] = keep
            else:
                running.pop(gpu_id, None)

        # Reap CPU fallback jobs (single slot)
        keep_cpu: List[RunningJob] = []
        for r in cpu_running:
            rc = r.process.poll()
            if rc is None:
                keep_cpu.append(r)
                continue
            elapsed = round(time.time() - r.start_time, 1)
            r.log_file_handle.write(f"\\nEND_AT: {datetime.now().isoformat()}\\n")
            r.log_file_handle.write(f"RETURN_CODE: {rc}\\n")
            r.log_file_handle.flush()
            r.log_file_handle.close()
            if rc == 0:
                completed.append(
                    {
                        "config": r.job.config,
                        "attempt": r.job.attempt,
                        "gpu_id": None,
                        "elapsed_sec": elapsed,
                        "log_path": str(r.log_path),
                    }
                )
                print(f"[DONE] {r.job.config} on CPU/device ({elapsed}s)")
            else:
                if r.job.attempt <= args.max_retries:
                    pending.append(Job(config=r.job.config, attempt=r.job.attempt + 1))
                    print(f"[RETRY] {r.job.config} failed on CPU/device (rc={rc})")
                else:
                    failed.append(
                        {
                            "config": r.job.config,
                            "attempt": r.job.attempt,
                            "gpu_id": None,
                            "return_code": rc,
                            "elapsed_sec": elapsed,
                            "log_path": str(r.log_path),
                        }
                    )
                    print(f"[FAIL] {r.job.config} on CPU/device (rc={rc})")
        cpu_running = keep_cpu

        write_state(state_path, pending, running, completed, failed)

        if stop_requested:
            break

        if not pending:
            time.sleep(args.poll_seconds)
            continue

        # Try GPU scheduling
        gpu_stats: List[Dict[str, int]] = []
        gpu_query_ok = False
        try:
            gpu_stats = query_gpus()
            gpu_query_ok = True
        except Exception as e:
            gpu_query_ok = False
            if not args.allow_cpu_fallback:
                print(f"[WARN] Cannot query GPUs via nvidia-smi: {e}")

        started_any = False
        if gpu_query_ok and gpu_stats:
            slots = available_gpu_slots(
                gpu_stats,
                running,
                min_free_mem_mb=args.min_free_mem_mb,
                max_gpu_util=args.max_gpu_util,
                max_jobs_per_gpu=args.max_jobs_per_gpu,
            )
            while pending and slots:
                gpu_id = slots.pop(0)
                job = pending.pop(0)
                rj = start_job(repo_root, args, run_log_dir, job, gpu_id=gpu_id, job_index=job_counter)
                job_counter += 1
                running.setdefault(gpu_id, []).append(rj)
                started_any = True
                print(
                    f"[START] {job.config} try={job.attempt} gpu={gpu_id} pid={getattr(rj.process, 'pid', None)} "
                    f"log={rj.log_path.name}"
                )

        # CPU fallback: run one at a time when enabled and no GPU job started this loop.
        if pending and args.allow_cpu_fallback and not started_any and not cpu_running:
            job = pending.pop(0)
            rj = start_job(repo_root, args, run_log_dir, job, gpu_id=None, job_index=job_counter)
            job_counter += 1
            cpu_running.append(rj)
            print(
                f"[START-CPU] {job.config} try={job.attempt} pid={getattr(rj.process, 'pid', None)} "
                f"log={rj.log_path.name}"
            )
            started_any = True

        if not started_any:
            if gpu_query_ok and gpu_stats:
                brief = ", ".join(
                    f"gpu{g['index']}:free={g['memory_free_mb']}MB,util={g['utilization']}%"
                    for g in gpu_stats
                )
                print(f"[WAIT] no available GPU slot ({brief}) | pending={len(pending)}")
            else:
                print(f"[WAIT] pending={len(pending)}")

        time.sleep(args.poll_seconds)

    if stop_requested:
        # Graceful shutdown: terminate all running jobs
        for gpu_jobs in running.values():
            for r in gpu_jobs:
                try:
                    r.process.terminate()
                except Exception:
                    pass
        for r in cpu_running:
            try:
                r.process.terminate()
            except Exception:
                pass
        print("Stop requested: sent terminate signal to running jobs.")

    write_state(state_path, pending, running, completed, failed)

    summary = {
        "completed": len(completed),
        "failed": len(failed),
        "pending": len(pending),
        "running": sum(len(v) for v in running.values()) + len(cpu_running),
        "log_dir": str(run_log_dir),
        "state_file": str(state_path),
    }
    print("\\nScheduler summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    return 0 if not failed and not pending else 2


if __name__ == "__main__":
    raise SystemExit(main())
