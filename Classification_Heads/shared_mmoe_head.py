"""Shared Multi-gate Mixture-of-Experts (MMoE) head for all tasks."""

import torch
import torch.nn as nn
from typing import Dict, List


class SharedMMoEHead(nn.Module):
    """
    Shared experts + per-task gates + per-task towers.
    Returns dict of {task_key: logits}.
    """

    def __init__(self, input_dim, task_keys, num_classes=3,
                 num_experts=4, expert_dim=512, tower_hidden_dim=512, dropout=0.1, **kwargs):
        super().__init__()
        self.task_keys = task_keys
        self.num_experts = num_experts

        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Dropout(dropout),
                nn.Linear(input_dim, expert_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_experts)
        ])

        self.task_gates = nn.ModuleDict({
            key: nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, num_experts),
            )
            for key in task_keys
        })

        self.task_towers = nn.ModuleDict({
            key: nn.Sequential(
                nn.LayerNorm(expert_dim),
                nn.Dropout(dropout),
                nn.Linear(expert_dim, tower_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(tower_hidden_dim, num_classes),
            )
            for key in task_keys
        })

    def forward(self, x) -> Dict[str, torch.Tensor]:
        expert_outputs = torch.stack([e(x) for e in self.shared_experts], dim=1)  # [B, E, D]
        outputs = {}
        for key in self.task_keys:
            gate_logits = self.task_gates[key](x)
            gate_weights = torch.softmax(gate_logits, dim=-1)  # [B, E]
            task_feat = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)  # [B, D]
            outputs[key] = self.task_towers[key](task_feat)
        return outputs
