# PES Stage-1 全要素消融实验

生成时间：2026-03-03 14:20:38

## 数据范围与指标

- 配置来源：`config/s1_*.py`（24 组架构）
- 结果来源：`outputs/config2.s1_*/run_*/training_history.json`
- 训练设定：每组 100 epochs，7 个 PES 子任务
- 本报告仅分析：`val_loss`、`val_auc`（按你的要求暂不讨论 ACC/F1）

## 图表索引

- Case 1（Concat+Linear，6 模型）：`case1_concat_linear_val_loss_auc.png`
- Case 2（严重过拟合，7 模型）：`case2_severe_overfit_val_loss_auc.png`
- Case 3A（BioMedCLIP 非平滑收敛，5 模型）：`case3_biomedclip_nonsmooth_val_loss_auc.png`
- Case 3B（CLIP 非平滑收敛，4 模型）：`case3_clip_nonsmooth_val_loss_auc.png`

## Case 1：无过拟合，AUC 稳定收敛（Concat + Linear）

| 模型 | Peak AUC (epoch) | Final AUC | ΔAUC(final-peak) | Min Val Loss (epoch) | Final Val Loss |
|---|---:|---:|---:|---:|---:|
| ResNet50 + 2 locals + concat + linear | 0.6646 (88) | 0.6588 | -0.0057 | 1.0183 (46) | 1.0325 |
| BioMedCLIP + 2 locals + concat + linear | 0.6594 (69) | 0.6577 | -0.0017 | 1.0267 (31) | 1.0775 |
| CLIP + 2 locals + concat + linear | 0.6908 (75) | 0.6907 | -0.0001 | 1.0144 (88) | 1.0146 |
| ResNet50 + 3 streams + concat + linear | 0.6655 (44) | 0.6576 | -0.0079 | 1.0082 (44) | 1.0326 |
| BioMedCLIP + 3 streams + concat + linear | 0.6585 (78) | 0.6580 | -0.0006 | 1.0619 (27) | 1.1489 |
| CLIP + 3 streams + concat + linear | 0.7009 (91) | 0.7004 | -0.0005 | 1.0017 (91) | 1.0027 |

- 组内统计：平均 Final AUC = 0.6705，平均 |Peak-Final| = 0.0027（取绝对值）
- 观察：该组整体曲线更平稳，AUC 后期波动较小，验证了 `Fusion=Concat + Head=Linear` 的稳定性规律。

## Case 2：严重过拟合，AUC 不收敛（崩盘/高噪声）

| 模型 | Peak AUC (epoch) | Final AUC | ΔAUC(final-peak) | Min Val Loss (epoch) | Final Val Loss |
|---|---:|---:|---:|---:|---:|
| ResNet50 + 2 locals + cross-attn + linear | 0.6694 (2) | 0.6171 | -0.0522 | 1.0149 (4) | 2.3778 |
| ResNet50 + 2 locals + cross-attn + mlp | 0.6826 (2) | 0.6184 | -0.0642 | 1.0002 (2) | 2.4686 |
| ResNet50 + 2 locals + concat + mlp | 0.6366 (74) | 0.6247 | -0.0119 | 1.0481 (1) | 1.9866 |
| ResNet50 + 3 streams + cross-attn + linear | 0.6720 (6) | 0.6344 | -0.0376 | 0.9744 (6) | 2.0945 |
| ResNet50 + 3 streams + cross-attn + mlp | 0.6360 (4) | 0.6061 | -0.0299 | 1.0594 (2) | 2.3934 |
| ResNet50 + 3 streams + concat + mlp | 0.6520 (2) | 0.6315 | -0.0205 | 1.0499 (1) | 2.0784 |
| BioMedCLIP + 2 locals + cross-attn + mlp | 0.6741 (22) | 0.6552 | -0.0189 | 0.9981 (7) | 1.7899 |

- 组内统计：平均 Final AUC = 0.6268，平均 `final_val_loss - min_val_loss` = 1.1492
- 观察：ResNet50 非 Concat+Linear 组合普遍出现早期峰值后退化，`val_loss` 后程抬升且 AUC 易崩盘或抖动。
- 特例：`BioMedCLIP + 2 locals + cross-attn + mlp`（s1_06）也落入该异常模式。

## Case 3：非平滑收敛（先达峰再回落收敛）

### BioMedCLIP 系列（5 模型）

| 模型 | Peak AUC (epoch) | Final AUC | ΔAUC(final-peak) | Min Val Loss (epoch) | Final Val Loss |
|---|---:|---:|---:|---:|---:|
| BioMedCLIP + 2 locals + cross-attn + linear | 0.6695 (45) | 0.6632 | -0.0063 | 1.0005 (9) | 1.4235 |
| BioMedCLIP + 2 locals + concat + mlp | 0.6760 (33) | 0.6617 | -0.0143 | 0.9699 (6) | 1.2251 |
| BioMedCLIP + 3 streams + cross-attn + linear | 0.6850 (32) | 0.6762 | -0.0088 | 1.0102 (10) | 1.4388 |
| BioMedCLIP + 3 streams + cross-attn + mlp | 0.6834 (8) | 0.6679 | -0.0155 | 0.9898 (7) | 1.8627 |
| BioMedCLIP + 3 streams + concat + mlp | 0.6770 (6) | 0.6735 | -0.0035 | 0.9741 (7) | 1.2783 |

### CLIP 系列（4 模型）

| 模型 | Peak AUC (epoch) | Final AUC | ΔAUC(final-peak) | Min Val Loss (epoch) | Final Val Loss |
|---|---:|---:|---:|---:|---:|
| CLIP + 2 locals + cross-attn + linear | 0.6925 (52) | 0.6818 | -0.0108 | 0.9661 (18) | 1.1059 |
| CLIP + 2 locals + concat + mlp | 0.6945 (9) | 0.6796 | -0.0148 | 0.9690 (15) | 1.1362 |
| CLIP + 3 streams + cross-attn + linear | 0.6988 (27) | 0.6890 | -0.0099 | 0.9632 (28) | 1.1098 |
| CLIP + 3 streams + cross-attn + mlp | 0.7195 (13) | 0.6869 | -0.0327 | 0.9780 (11) | 1.6592 |

- 组内统计：BioMedCLIP 平均 Peak AUC = 0.6782；CLIP 平均 Peak AUC = 0.7013
- 观察：大多数模型在约 30 epoch 左右达到低损失/高AUC窗口，随后出现回落并趋于新的收敛平台。

## 结论复核（与你给出的总结对齐）

1. 稳定性规律：`Concat + Linear` 组合在三种基座与两类输入规模下都更稳定，AUC 收敛性最好。
2. 基座差异：除 `Concat + Linear` 外，ResNet50 体系明显更易过拟合；BioMedCLIP 的 `2 locals + cross-attn + mlp` 为额外失稳特例。
3. 收敛模式：其余模型多体现“先峰值后回落”的非平滑收敛，峰值通常早于最终收敛值。

## 模型性能与 SOTA 选择

- 全部 24 组中最高 Peak AUC：`CLIP + 3 streams + cross-attn + mlp`，0.7195（epoch 13）
- 全部 24 组中最高 Final AUC：`CLIP + 3 streams + concat + linear`，0.7004

| 角色 | 模型 | Peak AUC | Final AUC | Min Val Loss (epoch) |
|---|---|---:|---:|---:|
| SOTA | CLIP + 3 streams + concat + linear | 0.7009 | 0.7004 | 1.0017 (91) |
| ResNet50 Baseline | ResNet50 + 2 locals + cross-attn + mlp | 0.6826 | 0.6184 | 1.0002 (2) |
| BioMedCLIP Baseline | BioMedCLIP + 2 locals + cross-attn + mlp | 0.6741 | 0.6552 | 0.9981 (7) |

- SOTA（CLIP + 3 streams + concat + linear）相对 ResNet50 baseline 的 Final AUC 提升：+0.0820
- SOTA 相对 BioMedCLIP baseline（2 locals/2 streams + cross-attn + mlp）的 Final AUC 提升：+0.0452

## 产物清单

- `analysis/case1_concat_linear_val_loss_auc.png`
- `analysis/case2_severe_overfit_val_loss_auc.png`
- `analysis/case3_biomedclip_nonsmooth_val_loss_auc.png`
- `analysis/case3_clip_nonsmooth_val_loss_auc.png`
- `analysis/stage1_metrics_summary.csv`
- `analysis/stage1_ablation_report.md`
