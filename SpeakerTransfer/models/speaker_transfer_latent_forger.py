import torch
from torch.nn import *
from .library import *


class LatentForgerModel(Module):
    def __init__(self):
        super().__init__()

        self.layers = Sequential(
            Conv1d(1024 + 2 * 128, 1024, 3, padding=1),
            LeakyReLU(negative_slope=0.1),
            Conv1d(1024, 1024, 3, padding=1),
            LeakyReLU(negative_slope=0.1),
            Conv1d(1024, 1536, 3, padding=1),
            LeakyReLU(negative_slope=0.1),
            Conv1d(1536, 1024, 3, padding=1),
            LeakyReLU(negative_slope=0.1),
            Conv1d(1024, 1024, 3, padding=1)
        )

    def forward(self, forgery_latent, forgery_categ, orig_categ):
        batch_size = forgery_latent.size()[0]
        forgery_latent = forgery_latent.reshape(batch_size, 1024, -1)
        _, _, width = forgery_latent.size()

        layer_input = torch.cat([
            forgery_latent,
            orig_categ.unsqueeze(dim=2).expand(-1, -1, width),
            forgery_categ.unsqueeze(dim=2).expand(-1, -1, width)
        ], dim=1)

        return self.layers(layer_input).reshape(batch_size, -1)


model = LatentForgerModel()
