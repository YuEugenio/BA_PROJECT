# Phase-2: LoRA on qkv input + attention output projections
from config.phase2_base_best import *
LORA_CONFIG = {
    'r': 16,
    'lora_alpha': 32,
    'target_modules': ['qkv_input', 'attn_output'],
    'lora_dropout': 0.1,
    'bias': 'none',
    'lora_variant': 'full',
}
