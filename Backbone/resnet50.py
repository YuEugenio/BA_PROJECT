"""ResNet50 backbone (ImageNet pretrained)."""

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def create_backbone(freeze=False, **kwargs):
    """
    Create ResNet50 backbone.
    Returns: (encoder: nn.Module, feature_dim: int, preprocess: callable)
    """
    weights = ResNet50_Weights.IMAGENET1K_V2
    backbone = resnet50(weights=weights)
    preprocess = weights.transforms()
    encoder = nn.Sequential(*list(backbone.children())[:-1])
    feature_dim = 2048

    if freeze:
        for p in encoder.parameters():
            p.requires_grad = False

    return encoder, feature_dim, preprocess


def extract_features(encoder, x):
    """Extract features and flatten."""
    import torch
    features = encoder(x)
    return torch.flatten(features, 1)
