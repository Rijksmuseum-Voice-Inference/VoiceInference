import torch


class Reconstructor(torch.nn.Module):
    def __init__(self, net, transform=None):
        super().__init__()
        self.net = net
        self.forward = self.reconst
        self.transform = transform

    def reconst(self, latent, metadata):
        return self.net(latent, metadata)

    def reconst_loss(self, reconst, original):
        if self.transform is not None:
            reconst = self.transform(reconst)
            original = self.transform(original)

        total = ((reconst - original) ** 2).sum()
        return total / reconst.numel()
