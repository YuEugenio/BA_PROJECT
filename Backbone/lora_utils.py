"""LoRA injection utilities for various backbone types."""

import math
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig

_QKV_INPUT_ALIASES = {'qkv_input', 'qkv_in', 'qkv', 'attn_qkv_input'}
_ATTN_OUTPUT_ALIASES = {'attn_output', 'attn_out', 'attention_output', 'proj'}
_MLP_ALIASES = {'mlp', 'mlp_ffn', 'mlp_head'}


def inject_lora(encoder, lora_config):
    """
    Inject LoRA adapters into a backbone encoder.

    Args:
        encoder: nn.Module backbone
        lora_config: dict with keys:
            - r: int (rank)
            - lora_alpha: int
            - target_modules: list of str
              Supports explicit module names and portable aliases:
              * qkv_input: attention QKV input projection
              * attn_output: attention output projection
              * mlp: transformer MLP projection layers
            - lora_dropout: float
            - bias: str
            - lora_variant: str ('full', 'qv_only', 'layer4_only', or None)

    Returns:
        encoder with LoRA injected
    """
    if lora_config is None:
        return encoder

    if 'target_modules' not in lora_config:
        raise ValueError("LORA_CONFIG must include 'target_modules'.")

    variant = lora_config.get('lora_variant', 'full')
    r = int(lora_config['r'])
    lora_alpha = int(lora_config['lora_alpha'])
    lora_dropout = float(lora_config.get('lora_dropout', 0.1))
    bias = lora_config.get('bias', 'none')

    resolved = _resolve_target_modules(encoder, lora_config['target_modules'])
    peft_targets = resolved['peft_targets']
    use_mha_qkv_input = resolved['use_mha_qkv_input']

    if use_mha_qkv_input:
        _inject_qkv_input_lora_for_mha(
            encoder,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            qv_only=(variant == 'qv_only'),
        )

    if peft_targets:
        peft_cfg = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=peft_targets,
            lora_dropout=lora_dropout,
            bias=bias,
        )
        encoder = get_peft_model(encoder, peft_cfg)

    if variant == 'qv_only':
        # Applies to PEFT LoRA layers whose output dim is fused [Q,K,V].
        _apply_qv_only_mask(encoder)
    elif variant == 'layer4_only':
        _freeze_non_layer4_lora(encoder)

    return encoder


def _resolve_target_modules(encoder, target_modules):
    """
    Resolve portable aliases to explicit targets, and validate matches.
    """
    if isinstance(target_modules, str):
        requested = [target_modules]
    else:
        requested = list(target_modules)

    if not requested:
        raise ValueError("LORA_CONFIG['target_modules'] cannot be empty.")

    module_names = [name for name, _ in encoder.named_modules()]
    has_mha_in_proj = _has_mha_in_proj_module(encoder)

    # Keep deterministic order while removing duplicates.
    seen = set()
    peft_targets = []
    use_mha_qkv_input = False
    for token in requested:
        expanded_targets, needs_mha_qkv = _expand_target_token(token, module_names, has_mha_in_proj)
        for t in expanded_targets:
            if t not in seen:
                seen.add(t)
                peft_targets.append(t)
        use_mha_qkv_input = use_mha_qkv_input or needs_mha_qkv

    def _target_match_count(target):
        # PEFT list matching is suffix-based in common usage; use suffix check
        # to validate that each effective target can hit at least one module.
        return sum(1 for name in module_names if name == target or name.endswith(f'.{target}') or name.endswith(target))

    missing = [t for t in peft_targets if _target_match_count(t) == 0]
    if missing:
        # Print a compact candidate list so users can configure explicit targets.
        candidates = sorted([
            name for name in module_names
            if (
                'attn.' in name or '.attn' in name or
                'qkv' in name or 'in_proj' in name or 'out_proj' in name or
                'mlp.' in name or '.mlp' in name or 'fc1' in name or 'fc2' in name
            )
        ])
        candidates = candidates[:40]
        raise ValueError(
            f"LoRA target_modules do not match encoder modules: {missing}. "
            f"Requested={requested}, Resolved={peft_targets}. "
            f"Use explicit module names from this backbone, e.g. candidates={candidates}"
        )

    if not peft_targets and not use_mha_qkv_input:
        raise ValueError(
            f"No valid LoRA targets were resolved from Requested={requested}."
        )

    return {
        'peft_targets': peft_targets,
        'use_mha_qkv_input': use_mha_qkv_input,
    }


def _expand_target_token(token, module_names, has_mha_in_proj):
    token = str(token).strip()
    if not token:
        return [], False

    if token in _QKV_INPUT_ALIASES:
        if _has_module_suffix(module_names, 'attn.qkv') or _has_module_suffix(module_names, 'qkv'):
            return ['qkv'], False
        if has_mha_in_proj:
            # CLIP ViT (open_clip) uses nn.MultiheadAttention.in_proj_weight.
            return [], True
        raise ValueError(
            f"Target alias '{token}' requested qkv input projection, but this backbone has "
            "neither an explicit qkv module nor MultiheadAttention.in_proj_weight."
        )

    if token in _ATTN_OUTPUT_ALIASES:
        if _has_module_suffix(module_names, 'attn.out_proj'):
            return ['attn.out_proj'], False
        if _has_module_suffix(module_names, 'attn.proj'):
            return ['attn.proj'], False
        raise ValueError(
            f"Target alias '{token}' requested attention output projection, but no matching "
            "attn.out_proj / attn.proj module was found."
        )

    if token in _MLP_ALIASES:
        if _has_module_suffix(module_names, 'mlp.c_fc') and _has_module_suffix(module_names, 'mlp.c_proj'):
            return ['mlp.c_fc', 'mlp.c_proj'], False
        if _has_module_suffix(module_names, 'mlp.fc1') and _has_module_suffix(module_names, 'mlp.fc2'):
            return ['mlp.fc1', 'mlp.fc2'], False
        if _has_module_suffix(module_names, 'fc1') and _has_module_suffix(module_names, 'fc2'):
            return ['fc1', 'fc2'], False
        raise ValueError(
            f"Target alias '{token}' requested MLP projection layers, but no matching "
            "MLP module pair was found."
        )

    # Cross-backbone compatibility for explicit fc aliases.
    if token == 'fc1':
        if _has_module_suffix(module_names, 'mlp.fc1'):
            return ['mlp.fc1'], False
        if _has_module_suffix(module_names, 'mlp.c_fc'):
            return ['mlp.c_fc'], False
    if token == 'fc2':
        if _has_module_suffix(module_names, 'mlp.fc2'):
            return ['mlp.fc2'], False
        if _has_module_suffix(module_names, 'mlp.c_proj'):
            return ['mlp.c_proj'], False

    # Cross-backbone compatibility for explicit projection alias.
    if token == 'proj':
        if _has_module_suffix(module_names, 'attn.out_proj'):
            return ['attn.out_proj'], False
        if _has_module_suffix(module_names, 'attn.proj'):
            return ['attn.proj'], False

    # Treat as explicit PEFT target module name.
    return [token], False


def _has_module_suffix(module_names, suffix):
    return any(name == suffix or name.endswith(f'.{suffix}') or name.endswith(suffix) for name in module_names)


def _has_mha_in_proj_module(encoder):
    for module in encoder.modules():
        if isinstance(module, nn.MultiheadAttention) and hasattr(module, 'in_proj_weight'):
            return True
    return False


def _inject_qkv_input_lora_for_mha(encoder, r, lora_alpha, lora_dropout, qv_only=False):
    if r <= 0:
        raise ValueError(f"LoRA rank r must be > 0, got {r}")

    injected = 0
    for module in encoder.modules():
        if not isinstance(module, nn.MultiheadAttention):
            continue
        if not hasattr(module, 'in_proj_weight'):
            continue
        if getattr(module, '_lora_qkv_input_injected', False):
            continue
        if not getattr(module, '_qkv_same_embed_dim', True):
            raise ValueError("qkv_input LoRA for nn.MultiheadAttention requires _qkv_same_embed_dim=True")

        in_dim = module.in_proj_weight.shape[1]
        out_dim = module.in_proj_weight.shape[0]
        dtype = module.in_proj_weight.dtype
        device = module.in_proj_weight.device

        lora_a = nn.Parameter(torch.empty((r, in_dim), dtype=dtype, device=device))
        lora_b = nn.Parameter(torch.zeros((out_dim, r), dtype=dtype, device=device))
        nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5))

        module.register_parameter('lora_qkv_A', lora_a)
        module.register_parameter('lora_qkv_B', lora_b)
        module.lora_qkv_scaling = float(lora_alpha) / float(r)
        module.lora_qkv_dropout_p = float(lora_dropout)
        module.lora_qkv_qv_only = bool(qv_only)
        module._lora_qkv_input_injected = True
        module.forward = types.MethodType(_mha_forward_with_qkv_lora, module)
        injected += 1

    if injected == 0:
        raise ValueError(
            "Requested qkv_input LoRA, but no nn.MultiheadAttention module with in_proj_weight was found."
        )


def _mha_forward_with_qkv_lora(
    self,
    query,
    key,
    value,
    key_padding_mask=None,
    need_weights=True,
    attn_mask=None,
    average_attn_weights=True,
    is_causal=False,
):
    is_batched = query.dim() == 3

    if self.batch_first and is_batched:
        if key is value:
            if query is key:
                query = key = value = query.transpose(1, 0)
            else:
                query, key = (x.transpose(1, 0) for x in (query, key))
                value = key
        else:
            query, key, value = (x.transpose(1, 0) for x in (query, key, value))

    delta = torch.matmul(self.lora_qkv_B, self.lora_qkv_A)
    if self.training and self.lora_qkv_dropout_p > 0.0:
        delta = F.dropout(delta, p=self.lora_qkv_dropout_p, training=True)
    if self.lora_qkv_qv_only and delta.shape[0] % 3 == 0:
        chunk = delta.shape[0] // 3
        delta = delta.clone()
        delta[chunk:2 * chunk, :] = 0

    in_proj_weight = self.in_proj_weight + (delta * self.lora_qkv_scaling).to(self.in_proj_weight.dtype)

    attn_output, attn_output_weights = F.multi_head_attention_forward(
        query=query,
        key=key,
        value=value,
        embed_dim_to_check=self.embed_dim,
        num_heads=self.num_heads,
        in_proj_weight=in_proj_weight,
        in_proj_bias=self.in_proj_bias,
        bias_k=self.bias_k,
        bias_v=self.bias_v,
        add_zero_attn=self.add_zero_attn,
        dropout_p=self.dropout,
        out_proj_weight=self.out_proj.weight,
        out_proj_bias=self.out_proj.bias,
        training=self.training,
        key_padding_mask=key_padding_mask,
        need_weights=need_weights,
        attn_mask=attn_mask,
        use_separate_proj_weight=False,
        average_attn_weights=average_attn_weights,
        is_causal=is_causal,
    )

    if self.batch_first and is_batched:
        return attn_output.transpose(1, 0), attn_output_weights
    return attn_output, attn_output_weights


def _apply_qv_only_mask(encoder):
    """
    Mask the K-channel portion of LoRA_B weights on fused QKV layers,
    effectively making LoRA only update Q and V.
    """
    for name, param in encoder.named_parameters():
        if 'lora_B' not in name:
            continue
        # qv_only should only apply to fused qkv input projection LoRA.
        # Do not mask attn_output / mlp LoRA even if their output dim % 3 == 0.
        if 'qkv' not in name:
            continue
        if param.ndim != 2:
            continue
        out_features = param.shape[0]
        if out_features % 3 != 0:
            continue

        chunk = out_features // 3
        with torch.no_grad():
            param[chunk: 2 * chunk, :].zero_()

        def _mask_k_grad(grad, chunk_size=chunk):
            grad = grad.clone()
            grad[chunk_size: 2 * chunk_size, :] = 0
            return grad

        param.register_hook(_mask_k_grad)


def _freeze_non_layer4_lora(encoder):
    """
    Keep LoRA trainable only on layer4 for sequential ResNet encoders.
    """
    lora_names = [name for name, _ in encoder.named_parameters() if 'lora_' in name]
    keep_names = [name for name in lora_names if '.7.' in name]

    if not keep_names:
        return

    for name, param in encoder.named_parameters():
        if 'lora_' not in name:
            continue
        if '.7.' not in name:
            param.requires_grad = False
