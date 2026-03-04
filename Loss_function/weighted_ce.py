"""Weighted cross-entropy loss."""

import numpy as np
import torch
import torch.nn as nn


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Compute inverse-frequency class weights for balanced CE loss."""
    unique, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    n_classes = len(unique)
    weights = total / (n_classes * counts)
    full_weights = np.ones(3)
    for cls, w in zip(unique, weights):
        full_weights[int(cls)] = w
    return torch.FloatTensor(full_weights)


def create_weighted_ce(class_weights_tensor, device):
    """Create a weighted CrossEntropyLoss."""
    return nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
