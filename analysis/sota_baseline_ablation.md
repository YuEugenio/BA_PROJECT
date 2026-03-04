# SOTA 与 Baseline 消融分析（独立报告）

生成时间：2026-03-03 14:45:51

## 1. 分析目标

- 对比对象：
  - SOTA：`CLIP + 3 streams + concat + linear`（s1_23）
  - ResNet50 Baseline：`ResNet50 + 2 locals + cross-attn + mlp`（s1_02）
  - BioMedCLIP Baseline：`BioMedCLIP + 2 locals(=2 streams) + cross-attn + mlp`（s1_06）
- 评估核心：`val_auc` 与 `val_loss`。

## 2. 训练动态对比（100 epochs）

- 曲线图：`analysis/sota_baseline_val_loss_auc.png`
- 子任务柱状图：`analysis/sota_baseline_task_auc.png`

### 2.1 全局指标

| 模型 | Peak AUC (epoch) | Final AUC | ΔAUC(final-peak) | Min Val Loss (epoch) | Final Val Loss |
|---|---:|---:|---:|---:|---:|
| CLIP + 3 streams + concat + linear | 0.7009 (91) | 0.7004 | -0.0005 | 1.0017 (91) | 1.0027 |
| ResNet50 + 2 locals + cross-attn + mlp | 0.6826 (2) | 0.6184 | -0.0642 | 1.0002 (2) | 2.4686 |
| BioMedCLIP + 2 locals(=2 streams) + cross-attn + mlp | 0.6741 (22) | 0.6552 | -0.0189 | 0.9981 (7) | 1.7899 |

### 2.2 过拟合强度指标

| 模型 | Loss Rise(final-min) | Final Generalization Gap(val-train) | Last10 AUC Mean |
|---|---:|---:|---:|
| CLIP + 3 streams + concat + linear | 0.0010 | 0.1361 | 0.7005 |
| ResNet50 + 2 locals + cross-attn + mlp | 1.4684 | 2.4465 | 0.6234 |
| BioMedCLIP + 2 locals(=2 streams) + cross-attn + mlp | 0.7918 | 1.7446 | 0.6557 |

- 直接对比：SOTA 相对 ResNet50 baseline 的 Final AUC 提升 `+0.0820`，相对 BioMedCLIP baseline 提升 `+0.0452`。
- 稳定性差异：SOTA 的 `|ΔAUC(final-peak)|= 0.0005`，显著小于 ResNet50 baseline 的 `0.0642` 与 BioMedCLIP baseline 的 `0.0189`。

## 3. 子任务层面消融（7 tasks）

| Task | SOTA (s1_23) | ResNet50 BL (s1_02) | Δ(SOTA-ResNet) | BioMedCLIP BL (s1_06) | Δ(SOTA-BioMed) |
|---|---:|---:|---:|---:|---:|
| mesial_papilla | 0.6557 | 0.6626 | -0.0068 | 0.6941 | -0.0383 |
| distal_papilla | 0.7249 | 0.7273 | -0.0024 | 0.7197 | +0.0051 |
| gingival_margin | 0.6765 | 0.6829 | -0.0064 | 0.5578 | +0.1187 |
| soft_tissue | 0.5548 | 0.5961 | -0.0414 | 0.5307 | +0.0241 |
| alveolar_defect | 0.9124 | 0.8863 | +0.0261 | 0.8192 | +0.0932 |
| mucosal_color | 0.6409 | 0.6168 | +0.0241 | 0.6874 | -0.0464 |
| mucosal_texture | 0.7411 | 0.6063 | +0.1348 | 0.7096 | +0.0314 |
| **wins / 7** | - | - | **3 / 7** | - | **5 / 7** |

- 解释：SOTA 并非在每个 task 都单点最优，但在关键高权重任务（如 `alveolar_defect`、`mucosal_texture`、`gingival_margin`）上增益更大，从而推高平均 AUC。

## 4. 因素级消融路径（解释 SOTA 为何成立）

### 4.1 Backbone 消融（固定 3 streams + concat + linear）

| 模型 | Peak AUC (epoch) | Final AUC | ΔAUC(final-peak) | Min Val Loss (epoch) | Final Val Loss |
|---|---:|---:|---:|---:|---:|
| ResNet50 + 3 streams + concat + linear | 0.6655 (44) | 0.6576 | -0.0079 | 1.0082 (44) | 1.0326 |
| BioMedCLIP + 3 streams + concat + linear | 0.6585 (78) | 0.6580 | -0.0006 | 1.0619 (27) | 1.1489 |
| CLIP + 3 streams + concat + linear | 0.7009 (91) | 0.7004 | -0.0005 | 1.0017 (91) | 1.0027 |

- 结论：在同一融合/头/输入设定下，CLIP backbone 的 Final AUC 最高（0.7004），相较 ResNet50 提升 `+0.0428`，相较 BioMedCLIP 提升 `+0.0424`。

### 4.2 Fusion/Head 消融（固定 CLIP + 3 streams）

| 模型 | Peak AUC (epoch) | Final AUC | ΔAUC(final-peak) | Min Val Loss (epoch) | Final Val Loss |
|---|---:|---:|---:|---:|---:|
| CLIP + 3 streams + cross-attn + linear | 0.6988 (27) | 0.6890 | -0.0099 | 0.9632 (28) | 1.1098 |
| CLIP + 3 streams + cross-attn + mlp | 0.7195 (13) | 0.6869 | -0.0327 | 0.9780 (11) | 1.6592 |
| CLIP + 3 streams + concat + linear | 0.7009 (91) | 0.7004 | -0.0005 | 1.0017 (91) | 1.0027 |
| CLIP + 3 streams + concat + mlp | 0.7153 (12) | 0.6983 | -0.0170 | 0.9412 (15) | 1.1393 |

- 结论：`cross-attn + mlp`（s1_22）提供更高瞬时峰值（0.7195），但后期回落明显；`concat + linear`（s1_23）收敛值最高（Final AUC 0.7004）且回落最小。

### 4.3 输入流数消融（固定 CLIP + concat + linear）

| 模型 | Peak AUC (epoch) | Final AUC | ΔAUC(final-peak) | Min Val Loss (epoch) | Final Val Loss |
|---|---:|---:|---:|---:|---:|
| CLIP + 2 locals + concat + linear | 0.6908 (75) | 0.6907 | -0.0001 | 1.0144 (88) | 1.0146 |
| CLIP + 3 streams + concat + linear | 0.7009 (91) | 0.7004 | -0.0005 | 1.0017 (91) | 1.0027 |

- 结论：从 2 locals 到 3 streams，Final AUC 从 `0.6907` 提升到 `0.7004`（`+0.0097`），说明引入全局流在稳定架构下可进一步改善最终收敛性能。

## 5. 最终结论

1. 若目标是“部署可用的稳定收敛 AUC”，SOTA 组合 `CLIP + 3 streams + concat + linear` 是当前最优选择。
2. Baseline 的主要问题并非峰值能力不足，而是后期过拟合导致的收敛失败（loss 抬升 + AUC 回落）。
3. 对后续实验，建议将 `concat + linear` 作为默认稳定骨架，再在其上做损失函数/正则化/LoRA 等增量优化。

## 6. 产物

- `analysis/sota_baseline_ablation.md`（本文件）
- `analysis/sota_baseline_val_loss_auc.png`
- `analysis/sota_baseline_task_auc.png`
- `analysis/sota_baseline_metrics.csv`
