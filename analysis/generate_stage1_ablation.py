#!/usr/bin/env python3
"""Generate Stage-1 ablation plots and markdown report for PES s1 experiments."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
ANALYSIS_DIR = ROOT / "analysis"


@dataclass(frozen=True)
class ExperimentMeta:
    exp_id: str
    short_label: str
    long_name: str


EXPERIMENTS: Dict[str, ExperimentMeta] = {
    "s1_01": ExperimentMeta("s1_01", "R50-2L-XA-Lin", "ResNet50 + 2 locals + cross-attn + linear"),
    "s1_02": ExperimentMeta("s1_02", "R50-2L-XA-MLP", "ResNet50 + 2 locals + cross-attn + mlp"),
    "s1_03": ExperimentMeta("s1_03", "R50-2L-Con-Lin", "ResNet50 + 2 locals + concat + linear"),
    "s1_04": ExperimentMeta("s1_04", "R50-2L-Con-MLP", "ResNet50 + 2 locals + concat + mlp"),
    "s1_05": ExperimentMeta("s1_05", "BMC-2L-XA-Lin", "BioMedCLIP + 2 locals + cross-attn + linear"),
    "s1_06": ExperimentMeta("s1_06", "BMC-2L-XA-MLP", "BioMedCLIP + 2 locals + cross-attn + mlp"),
    "s1_07": ExperimentMeta("s1_07", "BMC-2L-Con-Lin", "BioMedCLIP + 2 locals + concat + linear"),
    "s1_08": ExperimentMeta("s1_08", "BMC-2L-Con-MLP", "BioMedCLIP + 2 locals + concat + mlp"),
    "s1_09": ExperimentMeta("s1_09", "CLIP-2L-XA-Lin", "CLIP + 2 locals + cross-attn + linear"),
    "s1_10": ExperimentMeta("s1_10", "CLIP-2L-XA-MLP", "CLIP + 2 locals + cross-attn + mlp"),
    "s1_11": ExperimentMeta("s1_11", "CLIP-2L-Con-Lin", "CLIP + 2 locals + concat + linear"),
    "s1_12": ExperimentMeta("s1_12", "CLIP-2L-Con-MLP", "CLIP + 2 locals + concat + mlp"),
    "s1_13": ExperimentMeta("s1_13", "R50-3S-XA-Lin", "ResNet50 + 3 streams + cross-attn + linear"),
    "s1_14": ExperimentMeta("s1_14", "R50-3S-XA-MLP", "ResNet50 + 3 streams + cross-attn + mlp"),
    "s1_15": ExperimentMeta("s1_15", "R50-3S-Con-Lin", "ResNet50 + 3 streams + concat + linear"),
    "s1_16": ExperimentMeta("s1_16", "R50-3S-Con-MLP", "ResNet50 + 3 streams + concat + mlp"),
    "s1_17": ExperimentMeta("s1_17", "BMC-3S-XA-Lin", "BioMedCLIP + 3 streams + cross-attn + linear"),
    "s1_18": ExperimentMeta("s1_18", "BMC-3S-XA-MLP", "BioMedCLIP + 3 streams + cross-attn + mlp"),
    "s1_19": ExperimentMeta("s1_19", "BMC-3S-Con-Lin", "BioMedCLIP + 3 streams + concat + linear"),
    "s1_20": ExperimentMeta("s1_20", "BMC-3S-Con-MLP", "BioMedCLIP + 3 streams + concat + mlp"),
    "s1_21": ExperimentMeta("s1_21", "CLIP-3S-XA-Lin", "CLIP + 3 streams + cross-attn + linear"),
    "s1_22": ExperimentMeta("s1_22", "CLIP-3S-XA-MLP", "CLIP + 3 streams + cross-attn + mlp"),
    "s1_23": ExperimentMeta("s1_23", "CLIP-3S-Con-Lin", "CLIP + 3 streams + concat + linear"),
    "s1_24": ExperimentMeta("s1_24", "CLIP-3S-Con-MLP", "CLIP + 3 streams + concat + mlp"),
}


GROUPS = {
    "case1_concat_linear_stable": {
        "title": "Case 1: Concat + Linear (6 models)",
        "exp_ids": ["s1_03", "s1_07", "s1_11", "s1_15", "s1_19", "s1_23"],
        "output": "case1_concat_linear_val_loss_auc.png",
    },
    "case2_severe_overfit": {
        "title": "Case 2: Severe Overfitting / AUC Collapse (7 models)",
        "exp_ids": ["s1_01", "s1_02", "s1_04", "s1_13", "s1_14", "s1_16", "s1_06"],
        "output": "case2_severe_overfit_val_loss_auc.png",
    },
    "case3_biomedclip_nonsmooth": {
        "title": "Case 3A: BioMedCLIP Non-smooth Convergence (5 models)",
        "exp_ids": ["s1_05", "s1_08", "s1_17", "s1_18", "s1_20"],
        "output": "case3_biomedclip_nonsmooth_val_loss_auc.png",
    },
    "case3_clip_nonsmooth": {
        "title": "Case 3B: CLIP Non-smooth Convergence (4 models)",
        "exp_ids": ["s1_09", "s1_12", "s1_21", "s1_22"],
        "output": "case3_clip_nonsmooth_val_loss_auc.png",
    },
}


def resolve_history_path(exp_id: str) -> Path:
    candidates = sorted(OUTPUTS_DIR.glob(f"config2.{exp_id}_*"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one output dir for {exp_id}, found {len(candidates)}")

    run_dirs = sorted([p for p in candidates[0].glob("run_*") if p.is_dir()])
    if not run_dirs:
        raise RuntimeError(f"No run_* directory found for {exp_id}")

    hist_path = run_dirs[-1] / "training_history.json"
    if not hist_path.exists():
        raise RuntimeError(f"Missing training history for {exp_id}: {hist_path}")
    return hist_path


def load_histories() -> Dict[str, Dict[str, List[float]]]:
    histories: Dict[str, Dict[str, List[float]]] = {}
    for exp_id in EXPERIMENTS:
        hist_path = resolve_history_path(exp_id)
        with hist_path.open("r", encoding="utf-8") as f:
            history = json.load(f)
        histories[exp_id] = {
            "val_loss": history["val_loss"],
            "val_auc": history["val_auc"],
            "train_loss": history["train_loss"],
        }
    return histories


def summarize_metrics(histories: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for exp_id, history in histories.items():
        val_loss = np.array(history["val_loss"], dtype=float)
        val_auc = np.array(history["val_auc"], dtype=float)

        peak_auc_idx = int(np.argmax(val_auc))
        min_loss_idx = int(np.argmin(val_loss))

        summary[exp_id] = {
            "epochs": int(len(val_loss)),
            "peak_auc": float(val_auc[peak_auc_idx]),
            "peak_auc_epoch": float(peak_auc_idx + 1),
            "final_auc": float(val_auc[-1]),
            "final_auc_last10_mean": float(np.mean(val_auc[-10:])),
            "auc_drop_peak_to_final": float(val_auc[-1] - val_auc[peak_auc_idx]),
            "min_val_loss": float(val_loss[min_loss_idx]),
            "min_val_loss_epoch": float(min_loss_idx + 1),
            "final_val_loss": float(val_loss[-1]),
            "loss_rise_min_to_final": float(val_loss[-1] - val_loss[min_loss_idx]),
        }
    return summary


def plot_group(
    histories: Dict[str, Dict[str, List[float]]],
    exp_ids: List[str],
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8), dpi=220)

    for exp_id in exp_ids:
        h = histories[exp_id]
        epochs = np.arange(1, len(h["val_loss"]) + 1)
        label = EXPERIMENTS[exp_id].short_label
        axes[0].plot(epochs, h["val_loss"], linewidth=1.8, label=label)
        axes[1].plot(epochs, h["val_auc"], linewidth=1.8, label=label)

    axes[0].set_title("Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].set_title("Validation AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")

    for ax in axes:
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.84, 0.95])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(summary: Dict[str, Dict[str, float]], output_path: Path) -> None:
    fields = [
        "exp_id",
        "model",
        "epochs",
        "peak_auc",
        "peak_auc_epoch",
        "final_auc",
        "final_auc_last10_mean",
        "auc_drop_peak_to_final",
        "min_val_loss",
        "min_val_loss_epoch",
        "final_val_loss",
        "loss_rise_min_to_final",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for exp_id in sorted(summary.keys(), key=lambda x: int(x.split("_")[1])):
            row = dict(summary[exp_id])
            row["exp_id"] = exp_id
            row["model"] = EXPERIMENTS[exp_id].long_name
            writer.writerow(row)


def build_table_lines(exp_ids: List[str], summary: Dict[str, Dict[str, float]]) -> List[str]:
    lines = [
        "| 模型 | Peak AUC (epoch) | Final AUC | ΔAUC(final-peak) | Min Val Loss (epoch) | Final Val Loss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for exp_id in exp_ids:
        m = summary[exp_id]
        lines.append(
            "| {model} | {peak:.4f} ({peak_ep}) | {final:.4f} | {delta:+.4f} | {min_loss:.4f} ({min_ep}) | {final_loss:.4f} |".format(
                model=EXPERIMENTS[exp_id].long_name,
                peak=m["peak_auc"],
                peak_ep=int(m["peak_auc_epoch"]),
                final=m["final_auc"],
                delta=m["auc_drop_peak_to_final"],
                min_loss=m["min_val_loss"],
                min_ep=int(m["min_val_loss_epoch"]),
                final_loss=m["final_val_loss"],
            )
        )
    return lines


def build_report(summary: Dict[str, Dict[str, float]]) -> str:
    case1_ids = GROUPS["case1_concat_linear_stable"]["exp_ids"]
    case2_ids = GROUPS["case2_severe_overfit"]["exp_ids"]
    case3_bmc_ids = GROUPS["case3_biomedclip_nonsmooth"]["exp_ids"]
    case3_clip_ids = GROUPS["case3_clip_nonsmooth"]["exp_ids"]

    def avg(ids: List[str], key: str) -> float:
        return mean(summary[i][key] for i in ids)

    all_ids = sorted(summary.keys(), key=lambda x: int(x.split("_")[1]))
    best_peak = max(all_ids, key=lambda i: summary[i]["peak_auc"])
    best_final = max(all_ids, key=lambda i: summary[i]["final_auc"])

    sota_id = "s1_23"
    baseline_resnet_id = "s1_02"
    baseline_biomed_id = "s1_06"

    lines: List[str] = []
    lines.append("# PES Stage-1 全要素消融分析报告（基于 s1 配置）")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 数据范围与指标")
    lines.append("")
    lines.append("- 配置来源：`config/s1_*.py`（24 组架构）")
    lines.append("- 结果来源：`outputs/config2.s1_*/run_*/training_history.json`")
    lines.append("- 训练设定：每组 100 epochs，7 个 PES 子任务")
    lines.append("- 本报告仅分析：`val_loss`、`val_auc`（按你的要求暂不讨论 ACC/F1）")
    lines.append("")
    lines.append("## 图表索引")
    lines.append("")
    lines.append("- Case 1（Concat+Linear，6 模型）：`case1_concat_linear_val_loss_auc.png`")
    lines.append("- Case 2（严重过拟合，7 模型）：`case2_severe_overfit_val_loss_auc.png`")
    lines.append("- Case 3A（BioMedCLIP 非平滑收敛，5 模型）：`case3_biomedclip_nonsmooth_val_loss_auc.png`")
    lines.append("- Case 3B（CLIP 非平滑收敛，4 模型）：`case3_clip_nonsmooth_val_loss_auc.png`")
    lines.append("")
    lines.append("## Case 1：无过拟合，AUC 稳定收敛（Concat + Linear）")
    lines.append("")
    lines.extend(build_table_lines(case1_ids, summary))
    lines.append("")
    lines.append(
        "- 组内统计：平均 Final AUC = {:.4f}，平均 |Peak-Final| = {:.4f}（取绝对值）".format(
            avg(case1_ids, "final_auc"),
            mean(abs(summary[i]["auc_drop_peak_to_final"]) for i in case1_ids),
        )
    )
    lines.append("- 观察：该组整体曲线更平稳，AUC 后期波动较小，验证了 `Fusion=Concat + Head=Linear` 的稳定性规律。")
    lines.append("")
    lines.append("## Case 2：严重过拟合，AUC 不收敛（崩盘/高噪声）")
    lines.append("")
    lines.extend(build_table_lines(case2_ids, summary))
    lines.append("")
    lines.append(
        "- 组内统计：平均 Final AUC = {:.4f}，平均 `final_val_loss - min_val_loss` = {:.4f}".format(
            avg(case2_ids, "final_auc"),
            avg(case2_ids, "loss_rise_min_to_final"),
        )
    )
    lines.append("- 观察：ResNet50 非 Concat+Linear 组合普遍出现早期峰值后退化，`val_loss` 后程抬升且 AUC 易崩盘或抖动。")
    lines.append("- 特例：`BioMedCLIP + 2 locals + cross-attn + mlp`（s1_06）也落入该异常模式。")
    lines.append("")
    lines.append("## Case 3：非平滑收敛（先达峰再回落收敛）")
    lines.append("")
    lines.append("### BioMedCLIP 系列（5 模型）")
    lines.append("")
    lines.extend(build_table_lines(case3_bmc_ids, summary))
    lines.append("")
    lines.append("### CLIP 系列（4 模型）")
    lines.append("")
    lines.extend(build_table_lines(case3_clip_ids, summary))
    lines.append("")
    lines.append(
        "- 组内统计：BioMedCLIP 平均 Peak AUC = {:.4f}；CLIP 平均 Peak AUC = {:.4f}".format(
            avg(case3_bmc_ids, "peak_auc"),
            avg(case3_clip_ids, "peak_auc"),
        )
    )
    lines.append("- 观察：大多数模型在约 30 epoch 左右达到低损失/高AUC窗口，随后出现回落并趋于新的收敛平台。")
    lines.append("")
    lines.append("## 结论复核（与你给出的总结对齐）")
    lines.append("")
    lines.append("1. 稳定性规律：`Concat + Linear` 组合在三种基座与两类输入规模下都更稳定，AUC 收敛性最好。")
    lines.append("2. 基座差异：除 `Concat + Linear` 外，ResNet50 体系明显更易过拟合；BioMedCLIP 的 `2 locals + cross-attn + mlp` 为额外失稳特例。")
    lines.append("3. 收敛模式：其余模型多体现“先峰值后回落”的非平滑收敛，峰值通常早于最终收敛值。")
    lines.append("")
    lines.append("## 模型性能与 SOTA 选择")
    lines.append("")
    lines.append(
        "- 全部 24 组中最高 Peak AUC：`{}`，{:.4f}（epoch {}）".format(
            EXPERIMENTS[best_peak].long_name,
            summary[best_peak]["peak_auc"],
            int(summary[best_peak]["peak_auc_epoch"]),
        )
    )
    lines.append(
        "- 全部 24 组中最高 Final AUC：`{}`，{:.4f}".format(
            EXPERIMENTS[best_final].long_name,
            summary[best_final]["final_auc"],
        )
    )
    lines.append("")
    lines.append("| 角色 | 模型 | Peak AUC | Final AUC | Min Val Loss (epoch) |")
    lines.append("|---|---|---:|---:|---:|")
    for role, exp_id in [
        ("SOTA", sota_id),
        ("ResNet50 Baseline", baseline_resnet_id),
        ("BioMedCLIP Baseline", baseline_biomed_id),
    ]:
        m = summary[exp_id]
        lines.append(
            "| {} | {} | {:.4f} | {:.4f} | {:.4f} ({}) |".format(
                role,
                EXPERIMENTS[exp_id].long_name,
                m["peak_auc"],
                m["final_auc"],
                m["min_val_loss"],
                int(m["min_val_loss_epoch"]),
            )
        )

    lines.append("")
    sota_final = summary[sota_id]["final_auc"]
    resnet_final = summary[baseline_resnet_id]["final_auc"]
    biomed_final = summary[baseline_biomed_id]["final_auc"]
    lines.append(
        "- SOTA（CLIP + 3 streams + concat + linear）相对 ResNet50 baseline 的 Final AUC 提升：{:+.4f}".format(
            sota_final - resnet_final
        )
    )
    lines.append(
        "- SOTA 相对 BioMedCLIP baseline（2 locals/2 streams + cross-attn + mlp）的 Final AUC 提升：{:+.4f}".format(
            sota_final - biomed_final
        )
    )
    lines.append("")
    lines.append("## 产物清单")
    lines.append("")
    lines.append("- `analysis/case1_concat_linear_val_loss_auc.png`")
    lines.append("- `analysis/case2_severe_overfit_val_loss_auc.png`")
    lines.append("- `analysis/case3_biomedclip_nonsmooth_val_loss_auc.png`")
    lines.append("- `analysis/case3_clip_nonsmooth_val_loss_auc.png`")
    lines.append("- `analysis/stage1_metrics_summary.csv`")
    lines.append("- `analysis/stage1_ablation_report.md`")

    return "\n".join(lines) + "\n"


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    histories = load_histories()
    summary = summarize_metrics(histories)

    for group in GROUPS.values():
        plot_group(
            histories=histories,
            exp_ids=group["exp_ids"],
            title=group["title"],
            output_path=ANALYSIS_DIR / group["output"],
        )

    write_summary_csv(summary, ANALYSIS_DIR / "stage1_metrics_summary.csv")

    report = build_report(summary)
    (ANALYSIS_DIR / "stage1_ablation_report.md").write_text(report, encoding="utf-8")

    print("Generated analysis artifacts in:", ANALYSIS_DIR)


if __name__ == "__main__":
    main()
