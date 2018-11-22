from torch.nn import *
from .library import *

model = Sequential(
    Linear(256, 128),
    ReLU(),
    Linear(128, 64),
    ReLU(),
    Linear(64, 64),
    ReLU(),
    Linear(64, 1),
    Sigmoid()
)
