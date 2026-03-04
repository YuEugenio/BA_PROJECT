"""
Unified dataset builder for PES multi-task classification.
Handles data loading, ROI parsing, preprocessing, splitting, and dataloader creation.
"""

import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as T

from Task_Setting.task_catalog import resolve_tasks, get_label_columns


# ROI label mapping (LabelMe Chinese labels -> standard English names)
ROI_LABELS = {
    '种植牙': 'implant',
    '对侧牙': 'control',
    '上颌前牙': 'global',
}


def get_patient_name(folder_name: str) -> str:
    """Extract patient name from folder name (first token)."""
    parts = folder_name.strip().split()
    return parts[0] if parts else folder_name


def parse_labelme_json(json_path: str) -> Dict[str, List[float]]:
    """Parse LabelMe JSON to extract ROI bounding boxes."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rois = {}
    for shape in data.get('shapes', []):
        label = shape.get('label', '')
        shape_type = shape.get('shape_type', '')
        points = shape.get('points', [])

        if shape_type == 'rectangle' and len(points) == 2 and label in ROI_LABELS:
            x1, y1 = points[0]
            x2, y2 = points[1]
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            rois[ROI_LABELS[label]] = [x1, y1, x2, y2]

    return rois


def build_data_items(data_dir: str, label_file: str, task_keys: list) -> List[Dict]:
    """
    Build list of data items with labels for the specified tasks.

    Returns list of dicts: {'image_path', 'json_path', 'labels': [int,...], 'patient': str}
    """
    df = pd.read_excel(label_file)
    label_columns = get_label_columns(task_keys)

    label_map = {}
    for _, row in df.iterrows():
        image_id = str(row['图像']).strip()
        try:
            labels = [int(row[col]) for col in label_columns]
            label_map[image_id] = labels
        except (KeyError, ValueError):
            continue

    data_items = []
    for patient_folder in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, patient_folder)
        if not os.path.isdir(folder_path):
            continue

        patient_name = get_patient_name(patient_folder)

        for filename in sorted(os.listdir(folder_path)):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            image_id = os.path.splitext(filename)[0]
            json_path = os.path.join(folder_path, image_id + '.json')

            if os.path.exists(json_path) and image_id in label_map:
                data_items.append({
                    'image_path': os.path.join(folder_path, filename),
                    'json_path': json_path,
                    'labels': label_map[image_id],
                    'patient': patient_name,
                })

    return data_items


def build_train_preprocess(preprocess, enable_augment=True,
                           flip_prob=0.5, jitter_strength=0.05,
                           rotation_deg=8.0, blur_prob=0.1):
    """Wrap a preprocess function with training augmentations."""
    if not enable_augment:
        return preprocess

    aug = T.Compose([
        T.RandomHorizontalFlip(p=flip_prob),
        T.ColorJitter(brightness=jitter_strength, contrast=jitter_strength,
                      saturation=jitter_strength),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=blur_prob),
        T.RandomRotation(degrees=rotation_deg, fill=(0, 0, 0)),
    ])

    def _aug_preprocess(img):
        return preprocess(aug(img))

    return _aug_preprocess


def create_dataloaders(cfg, preprocess):
    """
    Create train and validation dataloaders from a config object.

    Args:
        cfg: config module with all settings
        preprocess: backbone preprocessing function

    Returns:
        (train_loader, val_loader, class_weights_dict, split_info)
    """
    import torch
    from Loss_function.weighted_ce import compute_class_weights

    data_items = build_data_items(cfg.DATA_DIR, cfg.LABEL_FILE, cfg.TASKS)
    print(f"Total samples: {len(data_items)}")

    label_columns = get_label_columns(cfg.TASKS)

    # Split
    split_mode = getattr(cfg, 'SPLIT_MODE', 'fixed')
    if split_mode == 'auto_search':
        from Split.auto_search import search_best_split
        train_items, val_items, split_info = search_best_split(
            data_items, label_columns,
            test_size=cfg.TEST_SIZE,
            seed_base=getattr(cfg, 'SPLIT_SEED_BASE', 42),
            num_trials=getattr(cfg, 'SPLIT_SEARCH_TRIALS', 500),
        )
        print(f"Auto split: seed={split_info['seed']}, score={split_info['score']:.4f}")
    else:
        from Split.patient_group import patient_group_split
        train_items, val_items = patient_group_split(
            data_items, cfg.TEST_SIZE, cfg.SPLIT_SEED
        )
        split_info = {'mode': 'fixed_seed', 'seed': cfg.SPLIT_SEED}

    print(f"Train: {len(train_items)}, Val: {len(val_items)}")

    # Class weights
    train_labels = np.array([item['labels'] for item in train_items])
    class_weights = {}
    for i, key in enumerate(cfg.TASKS):
        class_weights[key] = compute_class_weights(train_labels[:, i])

    # Preprocessing
    train_preprocess = build_train_preprocess(
        preprocess,
        enable_augment=getattr(cfg, 'TRAIN_AUGMENT', True),
        flip_prob=getattr(cfg, 'FLIP_PROB', 0.5),
        jitter_strength=getattr(cfg, 'JITTER_STRENGTH', 0.05),
        rotation_deg=getattr(cfg, 'ROTATION_DEG', 8.0),
        blur_prob=getattr(cfg, 'BLUR_PROB', 0.1),
    )

    # Create datasets based on input mode
    input_mode = cfg.INPUT_MODE
    if input_mode == 'two_local':
        from Input.two_local import TwoLocalDataset
        train_ds = TwoLocalDataset(train_items, train_preprocess, parse_labelme_json)
        val_ds = TwoLocalDataset(val_items, preprocess, parse_labelme_json)
    elif input_mode == 'two_local_one_global':
        from Input.two_local_one_global import TwoLocalOneGlobalDataset
        train_ds = TwoLocalOneGlobalDataset(train_items, train_preprocess, parse_labelme_json)
        val_ds = TwoLocalOneGlobalDataset(val_items, preprocess, parse_labelme_json)
    else:
        raise ValueError(f"Unknown input mode: {input_mode}")

    # Weighted sampler
    use_sampler = getattr(cfg, 'USE_WEIGHTED_SAMPLER', False)
    if use_sampler:
        strategy = getattr(cfg, 'SAMPLER_STRATEGY', 'max_task')
        label_freq = {key: np.bincount(train_labels[:, i], minlength=3)
                      for i, key in enumerate(cfg.TASKS)}
        inv_freq = {key: 1.0 / np.maximum(label_freq[key], 1) for key in cfg.TASKS}
        sample_weights = []
        for item in train_items:
            ws = [inv_freq[key][int(item['labels'][i])] for i, key in enumerate(cfg.TASKS)]
            if strategy == 'avg_task':
                sample_weights.append(float(np.mean(ws)))
            elif strategy == 'sum_task':
                sample_weights.append(float(np.sum(ws)))
            else:
                sample_weights.append(float(np.max(ws)))
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, sampler=sampler,
                                  num_workers=cfg.NUM_WORKERS, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                  num_workers=cfg.NUM_WORKERS, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, class_weights, split_info
