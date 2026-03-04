# LoRA tuning: larger rank.
from config.clipvit_3stream_concat_linear_attnprojlora.base import *

LORA_CONFIG = {
    'r': 32,
    'lora_alpha': 64,
    'target_modules': ['attn_output'],
    'lora_dropout': 0.1,
    'bias': 'none',
    'lora_variant': 'full',
}
