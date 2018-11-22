import torch


class LatentForger(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.forward = self.modify_latent

    def modify_latent(self, forgery_latent, forgery_categ, orig_categ):
        return self.net(
            forgery_latent, forgery_categ, orig_categ)

    def pretrain_loss(self, pretend_latent, orig_latent):
        total = ((pretend_latent - orig_latent) ** 2).sum()
        return total / pretend_latent.numel()
