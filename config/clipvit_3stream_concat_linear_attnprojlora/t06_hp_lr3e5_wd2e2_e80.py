# Hyperparameter tuning: lower LR, stronger weight decay, shorter schedule.
from config.clipvit_3stream_concat_linear_attnprojlora.base import *

LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.02
NUM_EPOCHS = 80
ETA_MIN_RATIO = 0.001
