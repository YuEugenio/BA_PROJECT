"""
model.py - 三路BioMedCLIP + LoRA模型模块
Triple-Stream BioMedCLIP Model with LoRA Fine-tuning

功能 / Features:
1. 加载BioMedCLIP Vision Encoder / Load BioMedCLIP Vision Encoder
2. 使用peft注入LoRA Adapter / Inject LoRA Adapter using peft
3. 三路共享权重特征提取 / Triple-stream shared-weight feature extraction
4. 4个独立分类头 / 4 independent classification heads
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
    
    架构 / Architecture:
    - 三路输入: Implant ROI, Control ROI, Global ROI
    - 共享BioMedCLIP Vision Encoder + LoRA
    - 特征融合: Concatenate → 1536维
    - 4个独立分类头: 各输出3类
    """
    
    def __init__(
        self,
        pretrained: str = BIOMEDCLIP_MODEL,
        lora_config: Dict = None,
        freeze_backbone: bool = True,
        device: str = 'cuda'
    ):
        """
        Args:
            pretrained: BioMedCLIP预训练权重路径
            lora_config: LoRA配置字典
            freeze_backbone: 是否冻结主干网络
            device: 设备
        """
        super().__init__()
        
        if lora_config is None:
            lora_config = LORA_CONFIG
        
        self.device = device
        self.feature_dim = 512  # BioMedCLIP ViT-B输出维度
        self.fused_dim = self.feature_dim * 3  # 三路拼接 = 1536
        
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
        
        Args:
            x: 图像张量 [B, 3, 224, 224]
            
        Returns:
            特征向量 [B, 512]
        """
        # BioMedCLIP vision encoder输出
        features = self.vision_encoder(x)
        # 如果输出是tuple，取第一个元素
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
        
        Args:
            implant: 种植牙ROI [B, 3, 224, 224]
            control: 对侧牙ROI [B, 3, 224, 224]  
            global_view: 上颌前牙整体ROI [B, 3, 224, 224]
            
        Returns:
            Dict[task_name, logits]: 各任务的输出logits [B, 3]
        """
        # 三路特征提取（共享权重）/ Triple-stream feature extraction (shared weights)
        feat_implant = self.extract_features(implant)    # [B, 512]
        feat_control = self.extract_features(control)    # [B, 512]
        feat_global = self.extract_features(global_view) # [B, 512]
        
        # 特征融合（拼接）/ Feature fusion (concatenation)
        fused = torch.cat([feat_implant, feat_control, feat_global], dim=-1)  # [B, 1536]
        
        # 4个分类头输出 / 4 classification head outputs
        outputs = {}
        for task_name, head in self.classification_heads.items():
            outputs[task_name] = head(fused)  # [B, 3]
        
        return outputs
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """
        获取可训练参数（LoRA + 分类头）
        Get trainable parameters (LoRA + classification heads)
        """
        params = []
        # LoRA参数
        for name, param in self.vision_encoder.named_parameters():
            if param.requires_grad:
                params.append(param)
        # 分类头参数
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
    
    Args:
        device: 设备
        pretrained: 预训练权重路径
        
    Returns:
        (model, preprocess)
    """
    model = PESMultiTaskModel(pretrained=pretrained, device=device)
    model = model.to(device)
    preprocess = model.get_preprocess()
    
    return model, preprocess


if __name__ == '__main__':
    # 最小可验证测试 / Minimal sanity check
    print("Testing model module...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 创建模型 / Create model
    model, preprocess = create_model(device=device)
    
    # 统计参数 / Count parameters
    params = model.count_parameters()
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Frozen parameters: {params['frozen']:,}")
    
    # 测试前向传播 / Test forward pass
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    with torch.no_grad():
        outputs = model(dummy_input, dummy_input, dummy_input)
    
    print(f"\nOutput shapes:")
    for task_name, logits in outputs.items():
        print(f"  {task_name}: {logits.shape}")
    
    print("\nModel module test completed!")
