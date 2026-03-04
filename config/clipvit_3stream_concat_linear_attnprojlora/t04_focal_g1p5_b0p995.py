# Focal-loss variant with less aggressive class-balanced weighting.
from config.clipvit_3stream_concat_linear_attnprojlora.base import *

LOSS_TYPE = 'cb_focal'
CB_FOCAL_GAMMA = 1.5
CB_FOCAL_BETA = 0.995
LEARNING_RATE = 5e-5
