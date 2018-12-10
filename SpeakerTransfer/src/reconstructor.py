import torch


class Reconstructor(torch.nn.Module):
    def __init__(self, net, exp_frac=None):
        super().__init__()
        self.net = net
        self.forward = self.reconst
        self.exp_frac = exp_frac

    def reconst(self, latent, metadata):
        return self.net(latent, metadata)

    def reconst_loss(self, reconst, original):
        reconst_exp = torch.exp(reconst)
        original_exp = torch.exp(original)

        total_exp = ((reconst_exp - original_exp) ** 2).sum()
        total_log = ((reconst - original) ** 2).sum()

        return (total_exp * self.exp_frac +
                total_log * (1.0 - self.exp_frac)) / reconst.numel()
