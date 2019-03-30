from torch.nn import *
from library import *


model = Sequential(
    PadToMinimum(45, 2),
    LearnableBias(),
    Conv1d(257, 256, 5),  # 69 -> 65
    LeakyReLU(negative_slope=0.1),
    Conv1d(256, 256, 5),  # 65 -> 61
    LeakyReLU(negative_slope=0.1),
    Conv1d(256, 256, 5),  # 61 -> 57
    LeakyReLU(negative_slope=0.1),
    Conv1d(256, 512, 5),  # 57 -> 53
    LeakyReLU(negative_slope=0.1),
    Conv1d(512, 1024, 5, stride=2),  # 53 -> 25
    LeakyReLU(negative_slope=0.1),
    Conv1d(1024, 1024, 5),  # 25 -> 21
    LeakyReLU(negative_slope=0.1),
    Conv1d(1024, 1536, 5, stride=2),  # 21 -> 9
    LeakyReLU(negative_slope=0.1),
    Conv1d(1536, 1536, 5),  # 9 -> 5
    LeakyReLU(negative_slope=0.1),
    Conv1d(1536, 1, 5),  # 5 -> 1
    GlobalAvgPool(),
)
