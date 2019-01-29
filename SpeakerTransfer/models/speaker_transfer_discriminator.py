from torch.nn import *
from library import *


model = Sequential(
    PadToMinimum(45, 2),
    LearnableBias(),
    Conv1d(257, 256, 5),  # 45 -> 41
    LeakyReLU(negative_slope=0.1),
    Conv1d(256, 256, 5),  # 41 -> 37
    LeakyReLU(negative_slope=0.1),
    Conv1d(256, 256, 5),  # 37 -> 33
    LeakyReLU(negative_slope=0.1),
    Conv1d(256, 512, 5),  # 33 -> 29
    LeakyReLU(negative_slope=0.1),
    Conv1d(512, 1024, 5, stride=2),  # 29 -> 13
    LeakyReLU(negative_slope=0.1),
    Conv1d(1024, 1024, 5, stride=2),  # 13 -> 5
    LeakyReLU(negative_slope=0.1),
    Conv1d(1024, 1, 5),  # 5 -> 1
    GlobalAvgPool(),
)
