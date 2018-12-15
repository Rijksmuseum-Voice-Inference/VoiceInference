from torch.nn import *
from library import *


class SpeakerTransferDescriber(Module):
    def __init__(self):
        super(SpeakerTransferDescriber, self).__init__()
        self.sizes = []
        self.layers = Sequential(
            AppendSize(self.sizes),
            PadToMinimum(93, 2),
            LearnableBias(),
            Conv1d(257, 512, 5, padding=2),
            LeakyReLU(negative_slope=0.1),
            Conv1d(512, 512, 5, padding=2),
            LeakyReLU(negative_slope=0.1),
            Conv1d(512, 512, 5, stride=2),  # 93 -> 45
            AppendSize(self.sizes),
            LeakyReLU(negative_slope=0.1),
            Conv1d(512, 512, 5, stride=2),  # 45 -> 21
            AppendSize(self.sizes),
            LeakyReLU(negative_slope=0.1),
            Dropout1d(),
            Conv1d(512, 1024, 5, stride=2),  # 21 -> 9
            AppendSize(self.sizes),
            LeakyReLU(negative_slope=0.1),
            Conv1d(1024, 1536, 5, stride=2),  # 9 -> 3
            LeakyReLU(negative_slope=0.1),
            Dropout1d(),
            Conv1d(1536, 1536, 3),  # 3 -> 1
            LeakyReLU(negative_slope=0.1),
            Conv1d(1536, 1536, 1),  # 1 -> 1
            LeakyReLU(negative_slope=0.1),
            Conv1d(1536, 1024 + 128, 1),  # 1 -> 1
            PartialAvgPool(128)
        )

    def forward(self, features):
        self.sizes *= 0
        result = self.layers(features)
        return (result, (self.sizes,))


model = SpeakerTransferDescriber()
