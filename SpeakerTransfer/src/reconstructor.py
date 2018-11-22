import torch


class Reconstructor(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.forward = self.reconst

    def reconst(self, latent, metadata):
        return self.net(latent, metadata)

    def reconst_loss(self, reconst, original):
        total = ((reconst - original) ** 2).sum()
        return total / reconst.numel()
