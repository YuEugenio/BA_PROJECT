# LoRA tuning: medium-small rank with stronger dropout.
from config.clipvit_3stream_concat_linear_attnprojlora.base import *

LORA_CONFIG = {
    'r': 8,
    'lora_alpha': 16,
    'target_modules': ['attn_output'],
    'lora_dropout': 0.2,
    'bias': 'none',
    'lora_variant': 'full',
}
