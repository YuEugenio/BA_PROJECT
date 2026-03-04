"""Bidirectional cross-attention fusion for two local streams, with optional global concat."""

import torch
import torch.nn as nn
from typing import List


class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention between two local streams.
    Optionally concatenates a global feature vector.
    """

    def __init__(self, feature_dim, attn_heads=8, dropout=0.1, **kwargs):
        super().__init__()
        self.cross_attn_ab = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=attn_heads,
            dropout=dropout, batch_first=True,
        )
        self.cross_attn_ba = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=attn_heads,
            dropout=dropout, batch_first=True,
        )
        self.feature_dim = feature_dim

    def forward(self, features_list):
        """
        Args:
            features_list: list of 2 or 3 tensors [B, D].
                - 2 tensors: bidirectional cross-attention, output = [attended_a; attended_b]
                - 3 tensors: cross-attention on first two, concat third (global)
        Returns:
            fused tensor [B, fused_dim]
        """
        feat_a = features_list[0].unsqueeze(1)  # [B, 1, D]
        feat_b = features_list[1].unsqueeze(1)  # [B, 1, D]

        attended_a, _ = self.cross_attn_ab(query=feat_a, key=feat_b, value=feat_b)
        attended_b, _ = self.cross_attn_ba(query=feat_b, key=feat_a, value=feat_a)

        parts = [attended_a.squeeze(1), attended_b.squeeze(1)]
        if len(features_list) >= 3:
            parts.append(features_list[2])

        return torch.cat(parts, dim=-1)

    def output_dim(self, input_dims):
        if len(input_dims) == 2:
            return self.feature_dim * 2
        return self.feature_dim * 2 + input_dims[2]
