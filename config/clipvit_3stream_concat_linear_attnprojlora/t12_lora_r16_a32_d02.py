# LoRA tuning: baseline rank with stronger dropout.
from config.clipvit_3stream_concat_linear_attnprojlora.base import *

LORA_CONFIG = {
    'r': 16,
    'lora_alpha': 32,
    'target_modules': ['attn_output'],
    'lora_dropout': 0.2,
    'bias': 'none',
    'lora_variant': 'full',
}
