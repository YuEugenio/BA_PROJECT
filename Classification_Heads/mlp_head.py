"""MLP classification head per task (LayerNorm + Linear + GELU + Linear)."""

import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(self, input_dim, num_classes=3, hidden_dim=512, dropout=0.1, **kwargs):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)
