# Legacy experiment: legacy/legacy_balanced (BioMedCLIP, 3-stream, concat, MLP head, QKV LoRA, CB-Focal, auto-search split, weighted sampler)
import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
LABEL_FILE = os.path.join(_PROJECT_ROOT, 'data', 'labels.xlsx')

TASKS = ['mesial_papilla', 'distal_papilla', 'soft_tissue', 'mucosal_color']
INPUT_MODE = 'two_local_one_global'
BACKBONE = 'biomedclip_vit'
FREEZE_BACKBONE = True
LORA_CONFIG = {
    'r': 16,
    'lora_alpha': 32,
    'target_modules': ['qkv'],
    'lora_dropout': 0.1,
    'bias': 'none',
    'lora_variant': 'full',
}
FUSION = 'concat'
HEAD_TYPE = 'mlp'
HEAD_HIDDEN_DIM = 512
HEAD_DROPOUT = 0.1
LOSS_TYPE = 'cb_focal'
CB_FOCAL_BETA = 0.999
CB_FOCAL_GAMMA = 1.5
USE_WEIGHTED_SAMPLER = True
SAMPLER_STRATEGY = 'max_task'
SPLIT_MODE = 'auto_search'
SPLIT_SEED_BASE = 42
SPLIT_SEARCH_TRIALS = 500
BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
NUM_WORKERS = 4
TEST_SIZE = 0.2
SPLIT_SEED = 42
DEVICE = 'cuda'
