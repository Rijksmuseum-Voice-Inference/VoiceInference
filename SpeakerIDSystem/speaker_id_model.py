import torch
from speaker_id import ResNet

MODEL_FILE = 'model.pt'


class SpeakerIDNet(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


model = SpeakerIDNet(ResNet(1, 4, 16, 64, 256, 1211))
