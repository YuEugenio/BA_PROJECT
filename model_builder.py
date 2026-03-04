"""
Unified model builder: assembles backbone + LoRA + fusion + classification heads from config.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class PESModel(nn.Module):
    """
    Configurable PES multi-task classification model.
    Supports any combination of backbone, LoRA, fusion, and head type.
    """

    def __init__(self, encoder, feature_dim, extract_fn, fusion, heads, input_mode):
        """
        Args:
            encoder: backbone nn.Module
            feature_dim: int, per-stream feature dimension
            extract_fn: callable(encoder, x) -> features
            fusion: nn.Module with forward(features_list) -> fused
            heads: nn.ModuleDict of {task_key: head_module} or a single module that returns dict
            input_mode: 'two_local' or 'two_local_one_global'
        """
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self._extract_fn = extract_fn
        self.fusion = fusion
        self.heads = heads
        self.input_mode = input_mode
        self._is_mmoe = not isinstance(heads, nn.ModuleDict)

    def extract_features(self, x):
        return self._extract_fn(self.encoder, x)

    def forward(self, *inputs):
        """
        Forward pass.
        For two_local: inputs = (implant, control)
        For two_local_one_global: inputs = (implant, control, global_view)
        """
        features = [self.extract_features(inp) for inp in inputs]
        fused = self.fusion(features)

        if self._is_mmoe:
            return self.heads(fused)
        else:
            return {key: head(fused) for key, head in self.heads.items()}

    def get_trainable_params(self):
        """Get all trainable parameters."""
        params = []
        for p in self.encoder.parameters():
            if p.requires_grad:
                params.append(p)
        if self.fusion is not None:
            for p in self.fusion.parameters():
                params.append(p)
        if self._is_mmoe:
            for p in self.heads.parameters():
                params.append(p)
        else:
            for p in self.heads.parameters():
                params.append(p)
        return params

    def save_model(self, path):
        state = {
            'encoder': self.encoder.state_dict(),
            'fusion': self.fusion.state_dict() if self.fusion else None,
            'heads': self.heads.state_dict(),
        }
        torch.save(state, path)
        print(f"Model saved to {path}")

    def load_model(self, path, device='cpu'):
        state = torch.load(path, map_location=device)
        self.encoder.load_state_dict(state['encoder'])
        if state.get('fusion') is not None and self.fusion is not None:
            self.fusion.load_state_dict(state['fusion'])
        self.heads.load_state_dict(state['heads'])
        print(f"Model loaded from {path}")

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable, 'frozen': total - trainable}


def build_model(cfg, device='cuda'):
    """
    Build a PESModel from config.

    Returns: (model, preprocess)
    """
    # 1. Create backbone
    backbone_type = cfg.BACKBONE
    if backbone_type == 'resnet18':
        from Backbone.resnet18 import create_backbone, extract_features
    elif backbone_type == 'resnet50':
        from Backbone.resnet50 import create_backbone, extract_features
    elif backbone_type == 'biomedclip_vit':
        from Backbone.biomedclip_vit import create_backbone, extract_features
    elif backbone_type == 'clip_vit':
        from Backbone.clip_vit import create_backbone, extract_features
    else:
        raise ValueError(f"Unknown backbone: {backbone_type}")

    freeze = getattr(cfg, 'FREEZE_BACKBONE', True)
    pretrained = getattr(cfg, 'PRETRAINED', None)
    encoder, feature_dim, preprocess = create_backbone(freeze=freeze, pretrained=pretrained)

    # 2. Inject LoRA if configured
    lora_config = getattr(cfg, 'LORA_CONFIG', None)
    if lora_config is not None:
        from Backbone.lora_utils import inject_lora
        encoder = inject_lora(encoder, lora_config)

    # 3. Create fusion module
    fusion_type = cfg.FUSION
    input_mode = cfg.INPUT_MODE
    if input_mode == 'two_local':
        n_streams = 2
    else:
        n_streams = 3

    if fusion_type == 'concat':
        from Feature_fusion.concat import ConcatFusion
        fusion = ConcatFusion()
        fused_dim = feature_dim * n_streams
    elif fusion_type == 'cross_attention':
        from Feature_fusion.cross_attention import CrossAttentionFusion
        attn_heads = getattr(cfg, 'ATTN_HEADS', 8)
        attn_dropout = getattr(cfg, 'ATTN_DROPOUT', 0.1)
        fusion = CrossAttentionFusion(feature_dim, attn_heads=attn_heads, dropout=attn_dropout)
        fused_dim = fusion.output_dim([feature_dim] * n_streams)
    else:
        raise ValueError(f"Unknown fusion: {fusion_type}")

    # 4. Create classification heads
    head_type = cfg.HEAD_TYPE
    task_keys = cfg.TASKS
    head_dropout = getattr(cfg, 'HEAD_DROPOUT', 0.1)
    head_hidden_dim = getattr(cfg, 'HEAD_HIDDEN_DIM', 512)

    if head_type == 'linear':
        from Classification_Heads.linear_head import LinearHead
        heads = nn.ModuleDict({
            key: LinearHead(fused_dim, num_classes=3)
            for key in task_keys
        })
    elif head_type == 'mlp':
        from Classification_Heads.mlp_head import MLPHead
        heads = nn.ModuleDict({
            key: MLPHead(fused_dim, num_classes=3, hidden_dim=head_hidden_dim, dropout=head_dropout)
            for key in task_keys
        })
    elif head_type == 'task_moe':
        from Classification_Heads.task_moe_head import MoEClassificationHead
        moe_cfg = getattr(cfg, 'MOE_CONFIG', {})
        heads = nn.ModuleDict({
            key: MoEClassificationHead(
                fused_dim, num_classes=3,
                num_experts=moe_cfg.get('num_experts', 4),
                hidden_dim=moe_cfg.get('hidden_dim', 512),
                dropout=moe_cfg.get('dropout', 0.1),
            )
            for key in task_keys
        })
    elif head_type == 'shared_mmoe':
        from Classification_Heads.shared_mmoe_head import SharedMMoEHead
        moe_cfg = getattr(cfg, 'MOE_CONFIG', {})
        heads = SharedMMoEHead(
            fused_dim, task_keys=task_keys, num_classes=3,
            num_experts=moe_cfg.get('num_experts', 4),
            expert_dim=moe_cfg.get('expert_dim', 512),
            tower_hidden_dim=moe_cfg.get('tower_hidden_dim', 512),
            dropout=moe_cfg.get('dropout', 0.1),
        )
    else:
        raise ValueError(f"Unknown head type: {head_type}")

    model = PESModel(encoder, feature_dim, extract_features, fusion, heads, input_mode)
    model = model.to(device)

    params = model.count_parameters()
    print(f"Model parameters - Total: {params['total']:,}, Trainable: {params['trainable']:,}, Frozen: {params['frozen']:,}")

    return model, preprocess
