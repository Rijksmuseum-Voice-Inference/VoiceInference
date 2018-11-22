import torch


class Forger(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.forward = self.forge

    def forge(self, orig, orig_categ, forgery_categ):
        return self.net(orig, orig_categ, forgery_categ)

    def pretrain_loss(self, forgery, orig):
        total = ((forgery - orig) ** 2).sum()
        return total / forgery.numel()
