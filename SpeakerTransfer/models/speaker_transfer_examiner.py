from torch.nn import *
from library import *


class SpeakerTransferExaminer(Module):
    def __init__(self):
        super(SpeakerTransferExaminer, self).__init__()

        self.layers = Sequential(
            PadToMinimum(45, 2),
            LearnableBias(),
            Conv1d(257 + 128, 256, 5),
            LeakyReLU(negative_slope=0.1),
            Conv1d(256, 256, 5),
            LeakyReLU(negative_slope=0.1),
            Conv1d(256, 256, 5),
            LeakyReLU(negative_slope=0.1),
            Conv1d(256, 512, 5),
            LeakyReLU(negative_slope=0.1),
            Conv1d(512, 1024, 5, stride=2),
            LeakyReLU(negative_slope=0.1),
            Conv1d(1024, 1024, 5, padding=2),
            LeakyReLU(negative_slope=0.1),
            Conv1d(1024, 1536, 5, stride=2),
            LeakyReLU(negative_slope=0.1),
            Conv1d(1536, 1536, 5, padding=2),
            LeakyReLU(negative_slope=0.1),
            Conv1d(1536, 1024, 5),
            GlobalAvgPool(),
        )

    def forward(self, features, categ):
        _, _, width = features.size()

        layer_input = torch.cat([
            features,
            categ.unsqueeze(dim=2).expand(-1, -1, width)
        ], dim=1)

        return self.layers(layer_input)


model = SpeakerTransferExaminer()
