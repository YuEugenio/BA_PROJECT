"""
model_legacy_taskgated.py - Legacy三路BioMedCLIP + LoRA + Task-specific Gated Fusion
"""

from typing import Dict, List, Tuple

import open_clip
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model


LORA_CONFIG = {
    'r': 16,
    'lora_alpha': 32,
    'target_modules': ['qkv'],
    'lora_dropout': 0.1,
    'bias': 'none',
}

BIOMEDCLIP_MODEL = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
PES_TASK_NAMES = ['近中牙龈乳头', '远中牙龈乳头', '软组织形态', '粘膜颜色']


class ClassificationHead(nn.Module):
    """
    分类头: 512维 → 3类
    """

    def __init__(self, input_dim: int = 512, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class TaskSpecificGate(nn.Module):
    """
    每个任务一套门控：输入三路拼接特征(1536)，输出三路权重(softmax)
    """

    def __init__(self, input_dim: int = 1536, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.gate(x), dim=-1)


class PESMultiTaskModel(nn.Module):
    def __init__(
        self,
        pretrained: str = BIOMEDCLIP_MODEL,
        lora_config: Dict = None,
        freeze_backbone: bool = True,
        device: str = 'cuda',
    ):
        super().__init__()

        if lora_config is None:
            lora_config = LORA_CONFIG

        self.device = device
        self.feature_dim = 512
        self.concat_dim = self.feature_dim * 3

        print(f'Loading BioMedCLIP from {pretrained}...')
        model, _, preprocess = open_clip.create_model_and_transforms(pretrained)
        self.preprocess = preprocess
        self.vision_encoder = model.visual

        if freeze_backbone:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        print('Injecting LoRA adapters...')
        peft_config = LoraConfig(
            r=lora_config['r'],
            lora_alpha=lora_config['lora_alpha'],
            target_modules=lora_config['target_modules'],
            lora_dropout=lora_config.get('lora_dropout', 0.1),
            bias=lora_config.get('bias', 'none'),
        )
        self.vision_encoder = get_peft_model(self.vision_encoder, peft_config)
        self.vision_encoder.print_trainable_parameters()

        self.task_gates = nn.ModuleDict({
            name: TaskSpecificGate(self.concat_dim)
            for name in PES_TASK_NAMES
        })

        self.classification_heads = nn.ModuleDict({
            name: ClassificationHead(self.feature_dim, num_classes=3)
            for name in PES_TASK_NAMES
        })

    def get_preprocess(self):
        return self.preprocess

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.vision_encoder(x)
        if isinstance(features, tuple):
            features = features[0]
        return features

    def forward(
        self,
        implant: torch.Tensor,
        control: torch.Tensor,
        global_view: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        feat_implant = self.extract_features(implant)
        feat_control = self.extract_features(control)
        feat_global = self.extract_features(global_view)

        concat_feature = torch.cat([feat_implant, feat_control, feat_global], dim=-1)

        outputs = {}
        for task_name in PES_TASK_NAMES:
            weights = self.task_gates[task_name](concat_feature)
            fused = (
                weights[:, 0:1] * feat_implant
                + weights[:, 1:2] * feat_control
                + weights[:, 2:3] * feat_global
            )
            outputs[task_name] = self.classification_heads[task_name](fused)

        return outputs

    def get_trainable_params(self) -> List[nn.Parameter]:
        params = []
        for _, param in self.vision_encoder.named_parameters():
            if param.requires_grad:
                params.append(param)
        params.extend(self.task_gates.parameters())
        params.extend(self.classification_heads.parameters())
        return params

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
        }

    def save_model(self, path: str):
        state_dict = {
            'vision_encoder': self.vision_encoder.state_dict(),
            'task_gates': self.task_gates.state_dict(),
            'classification_heads': self.classification_heads.state_dict(),
        }
        torch.save(state_dict, path)
        print(f'Model saved to {path}')

    def load_model(self, path: str):
        state_dict = torch.load(path, map_location=self.device)
        self.vision_encoder.load_state_dict(state_dict['vision_encoder'])
        if 'task_gates' in state_dict:
            self.task_gates.load_state_dict(state_dict['task_gates'])
        self.classification_heads.load_state_dict(state_dict['classification_heads'])
        print(f'Model loaded from {path}')


def create_model(device: str = 'cuda', pretrained: str = BIOMEDCLIP_MODEL) -> Tuple[PESMultiTaskModel, callable]:
    model = PESMultiTaskModel(pretrained=pretrained, device=device)
    model = model.to(device)
    preprocess = model.get_preprocess()
    return model, preprocess
