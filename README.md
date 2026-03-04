# PES Multi-Task Classification

A modular multi-task classification framework for PES (Pink Esthetic Score) evaluation.

This README reflects the **current experiment workflow**:
- Stage1: architecture search (24 configs)
- Stage2: LoRA search on the best Stage1 architecture (11 configs)
- Model selection priority: **AUC first**

## 1. Current Experiment Objective

The current protocol is:
1. Run Stage1 architecture grid and select the best architecture by validation `avg_auc`.
2. Fix that architecture in `config/phase2_base_best.py`.
3. Run Stage2 LoRA combinations on that fixed architecture.
4. Compare Stage2 by the same AUC-first rule.

## 2. Fixed Core Settings

Unless explicitly changed in config files, experiments use:
- Tasks: all 7 PES targets
- Split mode: `fixed`
- Split seed: `42`
- Test size: `0.2`
- Epochs: `100`
- Batch size: `8`
- LR: `1e-4`
- Weight decay: `0.01`
- Device: `cuda`

7 PES tasks:
- `mesial_papilla`
- `distal_papilla`
- `gingival_margin`
- `soft_tissue`
- `alveolar_defect`
- `mucosal_color`
- `mucosal_texture`

## 3. Stage1: Architecture Search (24 configs)

Stage1 explores the Cartesian product:
- Input mode (2): `two_local`, `two_local_one_global`
- Backbone (3): `resnet50`, `biomedclip_vit`, `clip_vit`
- Fusion (2): `cross_attention`, `concat`
- Head (2): `linear`, `mlp`

Total: `2 x 3 x 2 x 2 = 24` configs.

Config files are:
- `config/s1_01_...py` to `config/s1_24_...py`

## 4. Stage2: LoRA Search (11 configs)

Stage2 is based on `config/phase2_base_best.py` and evaluates 11 LoRA configurations.

LoRA primitive injection options:
- `qkv_input`
- `attn_output`
- `qv_only` (variant of qkv-input LoRA)
- `mlp`

Stage2 config list:
- Single (4):
  - `p2_01_lora_qkv_input`
  - `p2_02_lora_attn_output`
  - `p2_03_lora_qv_only`
  - `p2_04_lora_mlp`
- Pair (5):
  - `p2_05_lora_qkv_plus_attn_output`
  - `p2_07_lora_qkv_plus_mlp`
  - `p2_08_lora_qv_plus_attn_output`
  - `p2_09_lora_qv_plus_mlp`
  - `p2_10_lora_attn_output_plus_mlp`
- Triple (2):
  - `p2_06_lora_qkv_attn_output_mlp`
  - `p2_11_lora_qv_attn_output_mlp`

Optional control:
- `p2_00_no_lora` (not included in default Stage2 jobs queue)

## 5. Config Layout

All active Stage1/Stage2 configs are now directly under `config/`:
- `config/s1_*.py`
- `config/p2_*.py`
- `config/phase2_base_best.py`

The old `config/config2/` layout has been removed.

## 6. Run Commands

Single run examples:

```bash
python train.py --config s1_24_clipvit_3stream_concat_mlp
python train.py --config p2_01_lora_qkv_input
```

Stage1 scheduler:

```bash
nohup bash scripts/run_config2_stage1.sh > outputs/stage1_scheduler.out 2>&1 &
tail -f outputs/stage1_scheduler.out
```

Stage2 scheduler:

```bash
nohup bash scripts/run_config2_stage2.sh > outputs/stage2_scheduler.out 2>&1 &
tail -f outputs/stage2_scheduler.out
```

Queue dry-run (no execution):

```bash
python3 scripts/schedule_train.py --jobs-file scripts/config2_stage1_jobs.txt --list-only
python3 scripts/schedule_train.py --jobs-file scripts/config2_stage2_jobs.txt --list-only
```


## 7. Output Structure

Each run is saved to:

```text
outputs/<config_name>/run_YYYYMMDD_HHMMSS/
```

Key files:
- `config_snapshot.json`
- `best_model.pt`
- `best_results.json`
- `training_history.json`
- `training_curves.png`
- `confusion_matrices.png`
- `auc_curves.png`
