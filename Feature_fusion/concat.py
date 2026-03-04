"""Simple concatenation fusion."""

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """Concatenate feature vectors along the last dimension."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, features_list):
        """
        Args:
            features_list: list of tensors, each [B, D_i]
        Returns:
            [B, sum(D_i)]
        """
        return torch.cat(features_list, dim=-1)

    def output_dim(self, input_dims):
        return sum(input_dims)
