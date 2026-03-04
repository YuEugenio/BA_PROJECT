# Phase-1 config 12: input=two_local, backbone=clip_vit, fusion=concat, head=mlp, lora=none
import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
LABEL_FILE = os.path.join(_PROJECT_ROOT, 'data', 'labels.xlsx')

TASKS = ['mesial_papilla', 'distal_papilla', 'gingival_margin', 'soft_tissue', 'alveolar_defect', 'mucosal_color', 'mucosal_texture']
INPUT_MODE = 'two_local'
BACKBONE = 'clip_vit'
FREEZE_BACKBONE = True
LORA_CONFIG = None
FUSION = 'concat'
HEAD_TYPE = 'mlp'
HEAD_HIDDEN_DIM = 512
HEAD_DROPOUT = 0.1
LOSS_TYPE = 'weighted_ce'

BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
NUM_WORKERS = 4
TEST_SIZE = 0.2
SPLIT_SEED = 42
SPLIT_MODE = 'fixed'
DEVICE = 'cuda'
