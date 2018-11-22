import torch
from torch.nn import *
from .library import *


class SpeakerTransferForgerModel(Module):
    def __init__(self):
        super().__init__()
        self.sizes = []
        self.layers = Sequential(
            AppendSize(self.sizes),
            Conv1d(301 + 2 * 128, 512, 5, stride=2),
            AppendSize(self.sizes),
            LeakyReLU(negative_slope=0.1),
            Conv1d(512, 512, 5, stride=2),
            LeakyReLU(negative_slope=0.1),
            Conv1d(512, 512, 9, padding=4),
            LeakyReLU(negative_slope=0.1),
            Conv1d(512, 512, 9, padding=4),
            LeakyReLU(negative_slope=0.1),
            Conv1d(512, 512, 9, padding=4),
            LeakyReLU(negative_slope=0.1),
            Conv1d(512, 512, 9, padding=4),
            LeakyReLU(negative_slope=0.1),
            Conv1d(512, 512, 9, padding=4),
            LeakyReLU(negative_slope=0.1),
            ConvTranspose1d(512, 512, 5, stride=2),
            LeakyReLU(negative_slope=0.1),
            RevertSize(self.sizes),
            ConvTranspose1d(512, 512, 5, stride=2),
            LeakyReLU(negative_slope=0.1),
            RevertSize(self.sizes, transform={1: 512}),
            Conv1d(512, 512, 5, padding=2),
            LeakyReLU(negative_slope=0.1),
            Conv1d(512, 301, 5, padding=2)
        )

    def forward(self, orig, orig_categ, forgery_categ):
        self.sizes.clear()
        _, _, width = orig.size()

        layer_input = torch.cat([
            orig,
            orig_categ.unsqueeze(dim=2).expand(-1, -1, width),
            forgery_categ.unsqueeze(dim=2).expand(-1, -1, width)
        ], dim=1)

        return self.layers(layer_input)


model = SpeakerTransferForgerModel()
