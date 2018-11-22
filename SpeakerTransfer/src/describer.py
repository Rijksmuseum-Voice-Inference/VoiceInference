import torch


class Describer(torch.nn.Module):
    def __init__(self, net, num_categ):
        super().__init__()
        self.net = net
        self.num_categ = num_categ
        self.forward = self.describe

    def describe(self, values):
        descr, sizes = self.net(values)
        descr_size = descr.size()[1]
        (latent, categ) = torch.split(
            descr, [descr_size - self.num_categ, self.num_categ], dim=1)
        return (latent, sizes, categ)

    def latent(self, values):
        (latent, sizes, _) = self.forward(values)
        return (latent, sizes)

    def categ(self, values):
        (_, _, categ) = self.forward(values)
        return categ

    def categ_loss(self, predict_categ, true_categ):
        total = ((predict_categ - true_categ) ** 2).sum()
        return total / predict_categ.numel()

    def latent_loss(self, pretend_latent, orig_latent):
        total = ((pretend_latent - orig_latent) ** 2).sum()
        return total / pretend_latent.numel()
