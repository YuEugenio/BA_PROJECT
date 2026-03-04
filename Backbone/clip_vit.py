"""CLIP ViT backbone via open_clip (OpenAI CLIP or other variants)."""

import open_clip

# Default: OpenAI CLIP ViT-B/16
DEFAULT_CLIP_MODEL = 'ViT-B-16-quickgelu'
DEFAULT_CLIP_PRETRAINED = 'openai'


def create_backbone(freeze=True, pretrained=None, clip_model_name=None, **kwargs):
    """
    Create CLIP ViT backbone.
    Args:
        freeze: whether to freeze backbone weights
        pretrained: pretrained dataset name (e.g. 'openai') or hf-hub path
        clip_model_name: model architecture name (e.g. 'ViT-B-16-quickgelu')
    Returns: (encoder: nn.Module, feature_dim: int, preprocess: callable)
    """
    model_name = clip_model_name or DEFAULT_CLIP_MODEL
    pretrained_src = pretrained or DEFAULT_CLIP_PRETRAINED

    if pretrained_src.startswith('hf-hub:'):
        # HuggingFace hub model
        print(f"Loading CLIP ViT from {pretrained_src}...")
        model, _, preprocess = open_clip.create_model_and_transforms(pretrained_src)
    else:
        print(f"Loading CLIP ViT: {model_name} (pretrained={pretrained_src})...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_src
        )

    encoder = model.visual
    feature_dim = 512

    if freeze:
        for p in encoder.parameters():
            p.requires_grad = False

    return encoder, feature_dim, preprocess


def extract_features(encoder, x):
    """Extract features from ViT encoder."""
    features = encoder(x)
    if isinstance(features, tuple):
        features = features[0]
    return features
