from torch.nn import *
from .library import *
import numpy as np

BACKGROUND = np.log(0.01)

model = Sequential(
    PadToMinimum(189, 2, value=BACKGROUND),
    Conv1d(257, 512, 5, stride=2),  # 189 -> 93
    ReLU(),
    Conv1d(512, 512, 5, stride=2),  # 93 -> 45
    ReLU(),
    Conv1d(512, 1024, 5, stride=2),  # 45 -> 21
    ReLU(),
    Conv1d(1024, 1024, 5, stride=2),  # 21 -> 9
    ReLU(),
    Conv1d(1024, 512, 5),  # 9 -> 5
    ReLU(),
    Conv1d(512, 128, 5),  # 5 -> 1
    GlobalAvgPool()
)
