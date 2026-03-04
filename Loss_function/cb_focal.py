"""Class-Balanced Focal Loss."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CBFocalLoss(nn.Module):
    """Class-Balanced Focal Loss for multi-class classification."""

    def __init__(self, class_counts, beta=0.999, gamma=1.5, eps=1e-8):
        super().__init__()
        counts = np.asarray(class_counts, dtype=np.float64)
        counts = np.maximum(counts, 1.0)
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.maximum(effective_num, eps)
        weights = weights / np.maximum(weights.sum(), eps) * len(weights)
        self.register_buffer('class_weights', torch.tensor(weights, dtype=torch.float32))
        self.gamma = gamma

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        idx = torch.arange(targets.size(0), device=targets.device)
        target_log_probs = log_probs[idx, targets]
        target_probs = probs[idx, targets]
        alpha = self.class_weights[targets]
        focal_term = torch.pow(1.0 - target_probs, self.gamma)
        loss = -alpha * focal_term * target_log_probs
        return loss.mean()
