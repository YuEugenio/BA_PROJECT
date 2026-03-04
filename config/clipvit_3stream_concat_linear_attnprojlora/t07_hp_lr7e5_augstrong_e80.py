# Hyperparameter tuning: moderate LR with stronger augmentation.
from config.clipvit_3stream_concat_linear_attnprojlora.base import *

LEARNING_RATE = 7e-5
NUM_EPOCHS = 80
JITTER_STRENGTH = 0.10
ROTATION_DEG = 12.0
BLUR_PROB = 0.2
