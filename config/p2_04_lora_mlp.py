# Phase-2: LoRA on transformer MLP projections
from config.phase2_base_best import *
LORA_CONFIG = {
    'r': 16,
    'lora_alpha': 32,
    'target_modules': ['mlp'],
    'lora_dropout': 0.1,
    'bias': 'none',
    'lora_variant': 'full',
}
