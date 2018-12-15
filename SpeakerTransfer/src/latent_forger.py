import torch


class LatentForger(torch.nn.Module):
    def __init__(self, net):
        super(LatentForger, self).__init__()
        self.net = net
        self.forward = self.modify_latent

    def modify_latent_plus(self, orig_latent, orig_categ, forgery_categ):
        return self.net(orig_latent, orig_categ, forgery_categ)

    def modify_latent(self, orig_latent, orig_categ, forgery_categ):
        return self.modify_latent_plus(
            orig_latent, orig_categ, forgery_categ)[0]

    def pretrain_loss(self, pretend_latent, orig_latent):
        total = ((pretend_latent - orig_latent) ** 2).sum()
        return total / pretend_latent.numel()
