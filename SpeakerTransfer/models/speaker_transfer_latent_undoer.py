import torch
from torch.nn import *
from library import *


class LatentUndoerModel(Module):
    def __init__(self):
        super(LatentUndoerModel, self).__init__()

        self.layers = Sequential(
            Conv1d(1024 + 128, 1024, 1),
            LeakyReLU(negative_slope=0.1),
            Conv1d(1024, 1536, 1),
            LeakyReLU(negative_slope=0.1),
            Conv1d(1536, 1536, 1),
            LeakyReLU(negative_slope=0.1),
            Conv1d(1536, 1536, 1),
            LeakyReLU(negative_slope=0.1),
            Conv1d(1536, 1024, 1),
            LeakyReLU(negative_slope=0.1),
            Conv1d(1024, 1024, 1),
        )

    def forward(self, orig_latent, command):
        _, _, width = orig_latent.size()

        layer_input = torch.cat([
            orig_latent,
            command.unsqueeze(dim=2).expand(-1, -1, width),
        ], dim=1)

        return self.layers(layer_input)


model = LatentUndoerModel()
