from torch.nn import *
from .library import *

LOG_EPSILON = -10


model = Sequential(
    AddConst(-LOG_EPSILON),
    LearnableBias(),
    PadToMinimum(189, 2),
    Conv1d(258, 512, 5, stride=2),  # 189 -> 93
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
