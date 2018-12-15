from torch.nn import *
from library import *


class SpeakerTransferReconstructor(Module):
    def __init__(self):
        super(SpeakerTransferReconstructor, self).__init__()
        self.sizes = []
        self.layers = Sequential(
            ConvTranspose1d(1024, 1536, 3),
            LeakyReLU(negative_slope=0.1),
            ConvTranspose1d(1536, 1024, 5, stride=2),
            LeakyReLU(negative_slope=0.1),
            RevertSize(self.sizes),
            ConvTranspose1d(1024, 512, 5, stride=2),
            LeakyReLU(negative_slope=0.1),
            RevertSize(self.sizes),
            Transpose(1, 2),
            TupleSelector(GRU(
                512, 256, 2, bidirectional=True, batch_first=True), 0),
            Transpose(1, 2),
            ConvTranspose1d(512, 512, 5, stride=2),
            LeakyReLU(negative_slope=0.1),
            RevertSize(self.sizes),
            ConvTranspose1d(512, 512, 5, stride=2),
            LeakyReLU(negative_slope=0.1),
            RevertSize(self.sizes, transform={1: 512}),
            Conv1d(512, 257, 3, padding=1),
            LearnableBias(),
        )

    def forward(self, features, metadata):
        (sizes,) = metadata
        self.sizes *= 0
        self.sizes += sizes
        return self.layers(features)


model = SpeakerTransferReconstructor()
