# Base tuning config: p2_02 architecture + automatic seed search for balanced split.
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
LABEL_FILE = os.path.join(_PROJECT_ROOT, 'data', 'labels.xlsx')

TASKS = [
    'mesial_papilla',
    'distal_papilla',
    'gingival_margin',
    'soft_tissue',
    'alveolar_defect',
    'mucosal_color',
    'mucosal_texture',
]

INPUT_MODE = 'two_local_one_global'
BACKBONE = 'clip_vit'
FREEZE_BACKBONE = True

# p2_02 baseline LoRA position: attention output projection only.
LORA_CONFIG = {
    'r': 16,
    'lora_alpha': 32,
    'target_modules': ['attn_output'],
    'lora_dropout': 0.1,
    'bias': 'none',
    'lora_variant': 'full',
}

FUSION = 'concat'
HEAD_TYPE = 'linear'
LOSS_TYPE = 'weighted_ce'

BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
NUM_WORKERS = 4
TEST_SIZE = 0.2

# Automatic seed search to keep train/val class distributions close.
SPLIT_MODE = 'auto_search'
SPLIT_SEED_BASE = 42
SPLIT_SEARCH_TRIALS = 1000

# Default training augmentation knobs (can be overridden per config).
TRAIN_AUGMENT = True
FLIP_PROB = 0.5
JITTER_STRENGTH = 0.05
ROTATION_DEG = 8.0
BLUR_PROB = 0.1

DEVICE = 'cuda'
