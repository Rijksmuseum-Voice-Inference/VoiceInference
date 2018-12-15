import torch


class Discriminator(torch.nn.Module):
    def __init__(self, net):
        super(Discriminator, self).__init__()
        self.net = net
        self.forward = self.discriminate
        self.bce = torch.nn.BCEWithLogitsLoss()

    def discriminate(self, features, categ):
        return self.net(features, categ)

    def gen_loss(self, fake_decision):
        ones = fake_decision.new_ones(fake_decision.size())
        return self.bce(fake_decision, ones)

    def advers_loss(self, real_decision, fake_decision):
        zeros = real_decision.new_zeros(real_decision.size())
        ones = zeros + 1
        real_loss = self.bce(real_decision, ones)
        fake_loss = self.bce(fake_decision, zeros)
        return (real_loss + fake_loss) / 2.0
