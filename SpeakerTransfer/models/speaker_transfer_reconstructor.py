from torch.nn import *
from .library import *

EPSILON = 1e-6


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
            Conv1d(512, 301, 5, padding=2),
            ReLU()
        )

    def forward(self, features, metadata):
        (sizes, energy) = metadata
        self.sizes.clear()
        self.sizes += sizes
        result = self.layers(features)[:, :-1, :]

        energy_raw = torch.exp(energy) - 1.0
        result_raw = torch.exp(result) - 1.0
        result_energy_raw = torch.sqrt(
            (result_raw ** 2).mean(dim=1, keepdim=True))
        result_energy_raw = torch.clamp(result_energy_raw, min=EPSILON)
        result_raw = torch.cat([
            result_raw / result_energy_raw * energy_raw,
            energy_raw], dim=1)
        return torch.log(1.0 + result_raw)


model = SpeakerTransferReconstructor()
