"""Auto-search for optimal patient-level split across multiple seeds."""

import numpy as np
from typing import List, Dict, Tuple
from Split.patient_group import patient_group_split


def _task_class_counts(data_items, label_columns):
    """Count per-class samples for each task."""
    counts = {col: np.zeros(3, dtype=np.int64) for col in label_columns}
    for item in data_items:
        for i, col in enumerate(label_columns):
            counts[col][int(item['labels'][i])] += 1
    return counts


def _split_quality_score(train_items, val_items, label_columns,
                         min_val_per_class=1, min_train_per_class=1, penalty_weight=10.0):
    """Score a split: lower is better. Penalizes missing classes and distribution divergence."""
    train_counts = _task_class_counts(train_items, label_columns)
    val_counts = _task_class_counts(val_items, label_columns)

    divergence = 0.0
    penalty = 0.0
    counts_report = {}

    for col in label_columns:
        t_total = max(int(train_counts[col].sum()), 1)
        v_total = max(int(val_counts[col].sum()), 1)
        divergence += float(np.abs(train_counts[col] / t_total - val_counts[col] / v_total).sum())
        penalty += float(np.maximum(0, min_val_per_class - val_counts[col]).sum())
        penalty += float(np.maximum(0, min_train_per_class - train_counts[col]).sum())
        counts_report[col] = {
            'train_counts': train_counts[col].tolist(),
            'val_counts': val_counts[col].tolist(),
        }

    score = divergence + penalty_weight * penalty
    return score, {'divergence': divergence, 'penalty': penalty, 'score': score, 'counts': counts_report}


def search_best_split(data_items, label_columns, test_size=0.2,
                       seed_base=42, num_trials=500,
                       min_val_per_class=1, min_train_per_class=1):
    """Search over random seeds for the best balanced patient-level split."""
    if len(data_items) == 0:
        return [], [], {'mode': 'auto_search', 'seed': seed_base, 'score': None, 'trials': 0}

    best = None
    feasible_best = None
    feasible_count = 0

    for offset in range(num_trials):
        seed = seed_base + offset
        train_items, val_items = patient_group_split(data_items, test_size, seed)
        score, stats = _split_quality_score(
            train_items, val_items, label_columns,
            min_val_per_class, min_train_per_class,
        )
        candidate = {'seed': seed, 'score': score, 'train': train_items, 'val': val_items, 'stats': stats}

        if best is None or score < best['score']:
            best = candidate
        if stats['penalty'] == 0:
            feasible_count += 1
            if feasible_best is None or score < feasible_best['score']:
                feasible_best = candidate

    selected = feasible_best if feasible_best is not None else best
    info = {
        'mode': 'auto_search',
        'seed': selected['seed'],
        'score': selected['score'],
        'trials': num_trials,
        'feasible_trials': feasible_count,
        'split_stats': selected['stats'],
    }
    return selected['train'], selected['val'], info
