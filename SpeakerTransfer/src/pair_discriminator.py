import torch


class PairDiscriminator(torch.nn.Module):
    def __init__(self, examiner, distinguisher):
        super(PairDiscriminator, self).__init__()
        self.examiner = examiner
        self.distinguisher = distinguisher
        self.forward = self.discriminate
        self.bce = torch.nn.BCEWithLogitsLoss()

    def discriminate(self, categ, features1, features2):
        examined1 = self.examiner(features1, categ)
        examined2 = self.examiner(features2, categ)
        return self.distinguisher(examined1, examined2)

    def gen_loss(self, decision, truth):
        fake_truth = decision.new_zeros(decision.size()) + (1 - truth)
        return self.bce(decision, fake_truth)

    def advers_loss(self, decision, truth):
        truth = decision.new_zeros(decision.size()) + truth
        return self.bce(decision, truth)
