"""Single linear classification head per task."""

import torch.nn as nn


class LinearHead(nn.Module):
    def __init__(self, input_dim, num_classes=3, **kwargs):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)
