from torch.nn import *
from library import *


class SpeakerTransferDiscriminator(Module):
    def __init__(self):
        super(SpeakerTransferDiscriminator, self).__init__()

        self.layers = Sequential(
            PadToMinimum(45, 2),
            LearnableBias(),
            Conv1d(257 + 128, 256, 5),  # 45 -> 41
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
            Conv1d(1024, 1024, 5),  # 5 -> 1
            GlobalAvgPool(),
        )

    def forward(self, features, categ):
        _, _, width = features.size()

        layer_input = torch.cat([
            features,
            categ.unsqueeze(dim=2).expand(-1, -1, width)
        ], dim=1)

        return self.layers(layer_input)


model = SpeakerTransferDiscriminator()
