from torch.nn import *
from .library import *

EPSILON = 1e-6
LOG_EPSILON = -10


class SpeakerTransferReconstructor(Module):
    def __init__(self):
        super().__init__()
        self.sizes = []
        self.layers = Sequential(
            Reshape(512, -1),
            ConvTranspose1d(512, 1536, 1),
            LeakyReLU(negative_slope=0.1),
            ConvTranspose1d(1536, 1536, 1),
            LeakyReLU(negative_slope=0.1),
            ConvTranspose1d(1536, 1536, 3),
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
            Conv1d(512, 512, 5, padding=2),
            LeakyReLU(negative_slope=0.1),
            Conv1d(512, 257, 5, padding=2),
        )

    def forward(self, features, metadata):
        (sizes,) = metadata
        self.sizes.clear()
        self.sizes += sizes
        return self.layers(features)


model = SpeakerTransferReconstructor()
