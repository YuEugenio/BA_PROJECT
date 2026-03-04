# Phase-2: LoRA on qkv input + mlp projections
from config.phase2_base_best import *
LORA_CONFIG = {
    'r': 16,
    'lora_alpha': 32,
    'target_modules': ['qkv_input', 'mlp'],
    'lora_dropout': 0.1,
    'bias': 'none',
    'lora_variant': 'full',
}
