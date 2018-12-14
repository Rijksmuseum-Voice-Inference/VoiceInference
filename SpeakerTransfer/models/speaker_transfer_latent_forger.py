import torch
from torch.nn import *
from library import *


class LatentForgerModel(Module):
    def __init__(self):
        super().__init__()

        self.layers = Sequential(
            Conv1d(1024 + 2 * 128, 1536, 3, padding=1),
            LeakyReLU(negative_slope=0.1),
            Conv1d(1536, 1536, 3, padding=1),
            LeakyReLU(negative_slope=0.1),
            Transpose(1, 2),
            TupleSelector(GRU(
                1536, 512, 3, bidirectional=True, batch_first=True), 0),
            Transpose(1, 2),
            Conv1d(1024, 1024, 1),
            LeakyReLU(negative_slope=0.1),
            Conv1d(1024, 1024 + 1 + 128, 1),
            PartialAvgPool(1, 128)
        )

    def forward(self, orig_latent, orig_categ, forgery_categ):
        _, _, width = orig_latent.size()

        layer_input = torch.cat([
            orig_latent,
            orig_categ.unsqueeze(dim=2).expand(-1, -1, width),
            forgery_categ.unsqueeze(dim=2).expand(-1, -1, width)
        ], dim=1)

        return self.layers(layer_input)


model = LatentForgerModel()
