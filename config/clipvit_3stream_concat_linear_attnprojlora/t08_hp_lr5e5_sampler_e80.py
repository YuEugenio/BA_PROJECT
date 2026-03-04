# Hyperparameter tuning: weighted sampler to mitigate long-tail classes.
from config.clipvit_3stream_concat_linear_attnprojlora.base import *

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.02
NUM_EPOCHS = 80
USE_WEIGHTED_SAMPLER = True
SAMPLER_STRATEGY = 'avg_task'
