# Focal-loss variant with stronger focusing.
from config.clipvit_3stream_concat_linear_attnprojlora.base import *

LOSS_TYPE = 'cb_focal'
CB_FOCAL_GAMMA = 2.0
CB_FOCAL_BETA = 0.999
LEARNING_RATE = 5e-5
