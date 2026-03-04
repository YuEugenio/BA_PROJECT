"""Patient-level group split strategies."""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.model_selection import GroupShuffleSplit


def patient_group_split(data_items, test_size=0.2, random_state=42):
    """Split data ensuring all images from same patient stay together."""
    if len(data_items) == 0:
        return [], []
    groups = [item['patient'] for item in data_items]
    indices = np.arange(len(data_items))
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(indices, groups=groups))
    return [data_items[i] for i in train_idx], [data_items[i] for i in val_idx]
