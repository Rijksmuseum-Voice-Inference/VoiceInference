from torch.nn import *
from library import *


class SpeakerTransferReconstructor(Module):
    def __init__(self):
        super().__init__()
        self.sizes = []
        self.layers = Sequential(
            Reshape(1024, -1),
            ConvTranspose1d(1024, 1536, 3),
            LeakyReLU(negative_slope=0.1),
            ConvTranspose1d(1536, 1024, 5, stride=2),
            LeakyReLU(negative_slope=0.1),
            RevertSize(self.sizes),
            ConvTranspose1d(1024, 512, 5, stride=2),
            LeakyReLU(negative_slope=0.1),
            RevertSize(self.sizes),
            ConvTranspose1d(512, 512, 5, stride=2),
            LeakyReLU(negative_slope=0.1),
            RevertSize(self.sizes),
            ConvTranspose1d(512, 512, 5, stride=2),
            LeakyReLU(negative_slope=0.1),
            RevertSize(self.sizes, transform={1: 512}),
            Transpose(1, 2),
            TupleSelector(GRU(512, 512, 3, batch_first=True), 0),
            Transpose(1, 2),
            Slice(257, 1),
        )

    def forward(self, features, metadata):
        (sizes,) = metadata
        self.sizes.clear()
        self.sizes += sizes
        return self.layers(features)


model = SpeakerTransferReconstructor()
