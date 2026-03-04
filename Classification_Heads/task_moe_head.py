"""Task-specific Mixture-of-Experts classification head."""

import torch
import torch.nn as nn


class MoEClassificationHead(nn.Module):
    """Per-task MoE: each task has its own gate + expert pool."""

    def __init__(self, input_dim, num_classes=3, num_experts=4, hidden_dim=512, dropout=0.1, **kwargs):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, num_experts),
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Dropout(dropout),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        gate_weights = torch.softmax(self.gate(x), dim=-1)  # [B, E]
        expert_logits = torch.stack([e(x) for e in self.experts], dim=1)  # [B, E, C]
        return torch.sum(gate_weights.unsqueeze(-1) * expert_logits, dim=1)  # [B, C]
