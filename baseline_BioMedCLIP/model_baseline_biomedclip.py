"""
Baseline-style model: two-stream BioMedCLIP ViT + cross-attention + linear heads.
"""

from typing import Dict, List, Tuple

import open_clip
import torch
import torch.nn as nn


BIOMEDCLIP_MODEL = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
PES_TASK_NAMES = ['近中牙龈乳头', '远中牙龈乳头', '软组织形态', '粘膜颜色']


class LinearClassificationHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 3):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class PESBaselineBioMedCLIPModel(nn.Module):
    """
    Two-stream baseline-style architecture:
    - stream features by BioMedCLIP vision encoder
    - bidirectional cross-attention fusion
    - per-task single linear classifier
    """

    def __init__(
        self,
        pretrained: str = BIOMEDCLIP_MODEL,
        freeze_backbone: bool = False,
        attn_heads: int = 8,
        device: str = 'cuda',
    ):
        super().__init__()
        self.device = device

        print(f'Loading BioMedCLIP from {pretrained}...')
        model, _, preprocess = open_clip.create_model_and_transforms(pretrained)
        self.preprocess = preprocess
        self.vision_encoder = model.visual

        if freeze_backbone:
            for parameter in self.vision_encoder.parameters():
                parameter.requires_grad = False

        self.feature_dim = 512
        self.cross_attn_ab = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=attn_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.cross_attn_ba = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=attn_heads,
            dropout=0.1,
            batch_first=True,
        )

        fused_dim = self.feature_dim * 2
        self.classification_heads = nn.ModuleDict({
            task_name: LinearClassificationHead(fused_dim, num_classes=3)
            for task_name in PES_TASK_NAMES
        })

    def get_preprocess(self):
        return self.preprocess

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.vision_encoder(x)
        if isinstance(features, tuple):
            features = features[0]
        return features

    def forward(self, stream_a: torch.Tensor, stream_b: torch.Tensor) -> Dict[str, torch.Tensor]:
        feature_a = self.extract_features(stream_a)
        feature_b = self.extract_features(stream_b)

        token_a = feature_a.unsqueeze(1)
        token_b = feature_b.unsqueeze(1)

        attended_a, _ = self.cross_attn_ab(query=token_a, key=token_b, value=token_b)
        attended_b, _ = self.cross_attn_ba(query=token_b, key=token_a, value=token_a)

        fused = torch.cat([attended_a.squeeze(1), attended_b.squeeze(1)], dim=-1)

        outputs = {}
        for task_name, head in self.classification_heads.items():
            outputs[task_name] = head(fused)
        return outputs

    def get_trainable_params(self) -> List[nn.Parameter]:
        params = []
        for _, param in self.vision_encoder.named_parameters():
            if param.requires_grad:
                params.append(param)
        params.extend(self.cross_attn_ab.parameters())
        params.extend(self.cross_attn_ba.parameters())
        params.extend(self.classification_heads.parameters())
        return params

    def save_model(self, path: str):
        state_dict = {
            'vision_encoder': self.vision_encoder.state_dict(),
            'cross_attn_ab': self.cross_attn_ab.state_dict(),
            'cross_attn_ba': self.cross_attn_ba.state_dict(),
            'classification_heads': self.classification_heads.state_dict(),
        }
        torch.save(state_dict, path)
        print(f'Model saved to {path}')


def create_model(device: str = 'cuda', pretrained: str = BIOMEDCLIP_MODEL) -> Tuple[PESBaselineBioMedCLIPModel, callable]:
    model = PESBaselineBioMedCLIPModel(pretrained=pretrained, device=device)
    model = model.to(device)
    preprocess = model.get_preprocess()
    return model, preprocess

