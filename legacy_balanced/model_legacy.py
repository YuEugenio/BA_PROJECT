"""
model_legacy.py - 三路BioMedCLIP + LoRA模型模块（旧版）
Legacy triple-stream BioMedCLIP model with LoRA fine-tuning
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import open_clip
from peft import get_peft_model, LoraConfig, TaskType


# LoRA配置参数 / LoRA configuration parameters
LORA_CONFIG = {
    'r': 16,                      # LoRA秩 / LoRA rank
    'lora_alpha': 32,             # LoRA缩放因子 / LoRA scaling factor
    'target_modules': ['qkv'],     # 目标模块 / Target modules (Attention QKV)
    'lora_dropout': 0.1,          # Dropout
    'bias': 'none',               # 不训练bias / Don't train bias
}

# BioMedCLIP预训练权重 / BioMedCLIP pretrained weights
BIOMEDCLIP_MODEL = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'

# 4个PES分类任务名称 / 4 PES classification task names
PES_TASK_NAMES = ['近中牙龈乳头', '远中牙龈乳头', '软组织形态', '粘膜颜色']


class ClassificationHead(nn.Module):
    """
    分类头: 1536维 → 3类
    Classification Head: 1536-dim → 3 classes
    """
    def __init__(self, input_dim: int = 1536, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class PESMultiTaskModel(nn.Module):
    """
    PES多任务分类模型
    PES Multi-Task Classification Model
    """

    def __init__(
        self,
        pretrained: str = BIOMEDCLIP_MODEL,
        lora_config: Dict = None,
        freeze_backbone: bool = True,
        device: str = 'cuda'
    ):
        super().__init__()

        if lora_config is None:
            lora_config = LORA_CONFIG

        self.device = device
        self.feature_dim = 512
        self.fused_dim = self.feature_dim * 3

        # 加载BioMedCLIP模型 / Load BioMedCLIP model
        print(f"Loading BioMedCLIP from {pretrained}...")
        model, _, preprocess = open_clip.create_model_and_transforms(pretrained)
        self.preprocess = preprocess

        # 提取Vision Encoder / Extract Vision Encoder
        self.vision_encoder = model.visual

        # 冻结主干网络 / Freeze backbone
        if freeze_backbone:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        # 注入LoRA Adapter / Inject LoRA Adapter
        print("Injecting LoRA adapters...")
        peft_config = LoraConfig(
            r=lora_config['r'],
            lora_alpha=lora_config['lora_alpha'],
            target_modules=lora_config['target_modules'],
            lora_dropout=lora_config.get('lora_dropout', 0.1),
            bias=lora_config.get('bias', 'none'),
        )
        self.vision_encoder = get_peft_model(self.vision_encoder, peft_config)
        self.vision_encoder.print_trainable_parameters()

        # 4个分类头 / 4 classification heads
        self.classification_heads = nn.ModuleDict({
            name: ClassificationHead(self.fused_dim, num_classes=3)
            for name in PES_TASK_NAMES
        })

    def get_preprocess(self):
        """返回BioMedCLIP预处理函数 / Return BioMedCLIP preprocessing function"""
        return self.preprocess

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取单路图像特征
        Extract features from single stream
        """
        features = self.vision_encoder(x)
        if isinstance(features, tuple):
            features = features[0]
        return features

    def forward(
        self,
        implant: torch.Tensor,
        control: torch.Tensor,
        global_view: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        Forward pass
        """
        feat_implant = self.extract_features(implant)
        feat_control = self.extract_features(control)
        feat_global = self.extract_features(global_view)

        fused = torch.cat([feat_implant, feat_control, feat_global], dim=-1)

        outputs = {}
        for task_name, head in self.classification_heads.items():
            outputs[task_name] = head(fused)

        return outputs

    def get_trainable_params(self) -> List[nn.Parameter]:
        """
        获取可训练参数（LoRA + 分类头）
        Get trainable parameters (LoRA + classification heads)
        """
        params = []
        for name, param in self.vision_encoder.named_parameters():
            if param.requires_grad:
                params.append(param)
        for name, param in self.classification_heads.named_parameters():
            params.append(param)
        return params

    def count_parameters(self) -> Dict[str, int]:
        """
        统计参数量
        Count parameters
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }

    def save_model(self, path: str):
        """
        保存模型（仅保存可训练参数）
        Save model (only trainable parameters)
        """
        state_dict = {
            'vision_encoder': self.vision_encoder.state_dict(),
            'classification_heads': self.classification_heads.state_dict()
        }
        torch.save(state_dict, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """
        加载模型
        Load model
        """
        state_dict = torch.load(path, map_location=self.device)
        self.vision_encoder.load_state_dict(state_dict['vision_encoder'])
        self.classification_heads.load_state_dict(state_dict['classification_heads'])
        print(f"Model loaded from {path}")


def create_model(device: str = 'cuda', pretrained: str = BIOMEDCLIP_MODEL) -> Tuple[PESMultiTaskModel, callable]:
    """
    创建模型和预处理函数
    Create model and preprocessing function
    """
    model = PESMultiTaskModel(pretrained=pretrained, device=device)
    model = model.to(device)
    preprocess = model.get_preprocess()

    return model, preprocess
