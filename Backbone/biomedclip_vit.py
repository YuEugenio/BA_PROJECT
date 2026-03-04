"""BioMedCLIP ViT backbone via open_clip."""

import open_clip

BIOMEDCLIP_MODEL = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'


def create_backbone(freeze=True, pretrained=None, **kwargs):
    """
    Create BioMedCLIP ViT backbone.
    Returns: (encoder: nn.Module, feature_dim: int, preprocess: callable)
    """
    if pretrained is None:
        pretrained = BIOMEDCLIP_MODEL
    print(f"Loading BioMedCLIP from {pretrained}...")
    model, _, preprocess = open_clip.create_model_and_transforms(pretrained)
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
