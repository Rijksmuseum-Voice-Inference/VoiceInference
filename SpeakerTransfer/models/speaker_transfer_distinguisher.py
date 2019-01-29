from torch.nn import *
from library import *


class SpeakerTransferDistinguisher(Module):
    def __init__(self):
        super(SpeakerTransferDistinguisher, self).__init__()

        self.layers = Sequential(
            Linear(2048, 1024),
            ReLU(),
            Linear(1024, 512),
            ReLU(),
            Linear(512, 512),
            ReLU(),
            Linear(512, 128),
            ReLU(),
            Linear(128, 1),
        )

    def forward(self, left, right):
        combined = torch.cat([left, right], dim=1)
        return self.layers(combined)


model = SpeakerTransferDistinguisher()
