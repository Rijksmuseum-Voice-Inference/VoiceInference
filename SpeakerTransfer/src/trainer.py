#!/usr/bin/env python3

from __future__ import print_function
from util import print

import random
import numpy as np
import torch
import util
import IPython

import conversions
from loader import from_torch, to_torch

from describer import Describer
from reconstructor import Reconstructor
from latent_forger import LatentForger
from pair_discriminator import PairDiscriminator
from loader import VCTKLoader


DESCRIBER_FOOTER = "_describer"
RECONSTRUCTOR_FOOTER = "_reconstructor"
LATENT_FORGER_FOOTER = "_latent_forger"
DISCRIMINATOR_FOOTER = "_discriminator"
EXAMINER_FOOTER = "_examiner"
DISTINGUISHER_FOOTER = "_distinguisher"

example_tensor = torch.tensor(0.0)
if torch.cuda.is_available():
    try:
        example_tensor = example_tensor.cuda()
    except RuntimeError:
        pass

conversions
from_torch
to_torch


class Parameters:
    def __init__(self):
        self.data_path = "../VCTKProcessor/data/"
        self.speaker_categs_path = "../SpeakerFeatures/data/centers.pth"
        self.stage = ""
        self.header = "speaker_transfer"
        self.lr = 0.0001
        self.advers_lr = 0.00005
        self.categ_term = 0.10
        self.robustness_term = 0.20
        self.log_frac = 0.25
        self.advers_term = 0.05
        self.activity_term = 0.05
        self.undone_term = 0.05
        self.batch_size = 24
        self.num_periods = 0
        self.period_size = 10000
        self.rand_seed = -1
        self.sample_rate = 16000


parser = util.make_parser(Parameters(), "Latent Forger Model")


def train_analysts(params):
    speaker_categs = torch.load(params.speaker_categs_path)
    speaker_feature_dim = speaker_categs.size()[1]

    describer_model = util.load_model(params.header + DESCRIBER_FOOTER)
    describer = Describer(describer_model, speaker_feature_dim)
    util.initialize(describer)

    reconstructor_model = util.load_model(params.header + RECONSTRUCTOR_FOOTER)
    reconstructor = Reconstructor(reconstructor_model, params.log_frac)
    util.initialize(reconstructor)

    examiner_model = util.load_model(params.header + EXAMINER_FOOTER)
    distinguisher_model = util.load_model(params.header + DISTINGUISHER_FOOTER)
    discriminator = PairDiscriminator(examiner_model, distinguisher_model)
    util.initialize(discriminator)
    discriminator.train()

    if example_tensor.is_cuda:
        speaker_categs = speaker_categs.cuda()
        describer = describer.cuda()
        reconstructor = reconstructor.cuda()
        discriminator = discriminator.cuda()

    optim = torch.optim.Adam(
        torch.nn.Sequential(describer, reconstructor).parameters(),
        lr=params.lr)
    advers_optim = torch.optim.Adam(
        discriminator.parameters(),
        lr=params.advers_lr)

    data_loader = VCTKLoader(
        params.data_path, example_tensor, features='log')
    data_iterator = iter(data_loader)

    categ_loss_sum_batch = 0.0
    robustness_loss_sum_batch = 0.0
    reconst_loss_sum_batch = 0.0
    gen_loss_batch = 0.0
    advers_loss_batch = 0.0

    loss_sum_batch = 0.0
    discrim_loss_sum_batch = 0.0
    num_in_batch = 0

    period = 0
    while period < params.num_periods:
        period += 1

        loss_sum = 0.0
        loss_count = 0

        print(util.COMMENT_HEADER, end='')

        for _ in range(params.period_size):
            orig, orig_speaker = next(data_iterator)
            orig_categ = speaker_categs[orig_speaker].unsqueeze(0)

            describer.eval()
            center_categ = describer.categ(orig).detach()
            describer.train()

            (latent, metadata, pred_categ) = describer.describe(orig)
            reconst = reconstructor.reconst(latent, metadata)

            truth = 0 if (np.random.random() < 0.5) else 1

            if truth == 0:
                decision = discriminator.discriminate(
                    orig_categ, orig, reconst)
            else:
                decision = discriminator.discriminate(
                    orig_categ, reconst, orig)

            categ_loss = describer.categ_loss(pred_categ, orig_categ)
            robustness_loss = describer.categ_loss(pred_categ, center_categ)
            reconst_loss = reconstructor.reconst_loss(reconst, orig)
            gen_loss = discriminator.gen_loss(decision, truth)
            advers_loss = discriminator.advers_loss(decision, truth)

            loss = (
                params.categ_term * categ_loss +
                params.robustness_term * robustness_loss +
                reconst_loss +
                params.advers_term * gen_loss)
            discrim_loss = advers_loss

            categ_loss_sum_batch += categ_loss.item()
            robustness_loss_sum_batch += robustness_loss.item()
            reconst_loss_sum_batch += reconst_loss.item()
            gen_loss_batch += gen_loss.item()
            advers_loss_batch += advers_loss.item()

            loss_sum_batch = loss_sum_batch + loss
            discrim_loss_sum_batch = discrim_loss_sum_batch + discrim_loss
            num_in_batch += 1

            if num_in_batch >= params.batch_size:
                mean_discrim_loss = discrim_loss_sum_batch / num_in_batch
                if gen_loss_batch / num_in_batch <= 10.0:
                    advers_optim.zero_grad()
                    mean_discrim_loss.backward(retain_graph=True)
                    advers_optim.step()

                mean_loss = loss_sum_batch / num_in_batch
                optim.zero_grad()
                mean_loss.backward()
                optim.step()

                print("(" + "|".join([
                    "%0.3f" % (categ_loss_sum_batch / num_in_batch),
                    "%0.3f" % (robustness_loss_sum_batch / num_in_batch),
                    "%0.3f" % (reconst_loss_sum_batch / num_in_batch),
                    "%0.3f" % (gen_loss_batch / num_in_batch),
                    "%0.3f" % (advers_loss_batch / num_in_batch)]) + ")",
                    end=' ', flush=True)

                categ_loss_sum_batch = 0.0
                robustness_loss_sum_batch = 0.0
                reconst_loss_sum_batch = 0.0
                gen_loss_batch = 0.0
                advers_loss_batch = 0.0

                loss_sum_batch = 0.0
                discrim_loss_sum_batch = 0.0
                num_in_batch = 0

            loss_sum += loss.item()
            loss_count += 1

        print('')
        loss_mean = loss_sum / loss_count

        metrics = [
            ('period', period),
            ('loss', round(loss_mean, 3))
        ]
        util.print_metrics(metrics)

        torch.save(
            describer.state_dict(),
            'snapshots/' + params.header + DESCRIBER_FOOTER + '.pth')

        torch.save(
            reconstructor.state_dict(),
            'snapshots/' + params.header + RECONSTRUCTOR_FOOTER + '.pth')


def pretrain_manipulators(params):
    speaker_categs = torch.load(params.speaker_categs_path)
    num_speakers, speaker_feature_dim = speaker_categs.size()

    describer_model = util.load_model(params.header + DESCRIBER_FOOTER)
    describer = Describer(describer_model, speaker_feature_dim)
    describer.load_state_dict(torch.load(
        'snapshots/' + params.header + DESCRIBER_FOOTER + '.pth'))
    describer.eval()

    reconstructor_model = util.load_model(params.header + RECONSTRUCTOR_FOOTER)
    reconstructor = Reconstructor(reconstructor_model, params.log_frac)
    reconstructor.load_state_dict(torch.load(
        'snapshots/' + params.header + RECONSTRUCTOR_FOOTER + '.pth'))

    latent_forger_model = util.load_model(params.header + LATENT_FORGER_FOOTER)
    latent_forger = LatentForger(latent_forger_model)
    util.initialize(latent_forger)

    if example_tensor.is_cuda:
        speaker_categs = speaker_categs.cuda()
        describer = describer.cuda()
        reconstructor = reconstructor.cuda()
        latent_forger = latent_forger.cuda()

    optim = torch.optim.Adam(latent_forger.parameters(), lr=params.lr)

    data_loader = VCTKLoader(
        params.data_path, example_tensor, features='log')
    data_iterator = iter(data_loader)

    latent_loss_sum_batch = 0.0
    reconst_loss_sum_batch = 0.0
    loss_sum_batch = 0.0
    num_in_batch = 0

    period = 0
    while period < params.num_periods:
        period += 1

        loss_sum = 0.0
        loss_count = 0

        print(util.COMMENT_HEADER, end='')

        for _ in range(params.period_size):
            orig, orig_speaker = next(data_iterator)
            orig_categ = speaker_categs[orig_speaker].unsqueeze(0)

            forgery_categ = speaker_categs[
                np.random.randint(num_speakers)].unsqueeze(0)

            orig_latent, metadata = describer.latent(orig)
            orig_latent = orig_latent.detach()
            pretend_latent = latent_forger.modify_latent(
                orig_latent, forgery_categ, orig_categ)
            pretend_reconst = reconstructor.reconst(pretend_latent, metadata)

            latent_loss = latent_forger.pretrain_loss(
                pretend_latent, orig_latent)
            reconst_loss = reconstructor.reconst_loss(pretend_reconst, orig)

            loss = latent_loss + reconst_loss

            latent_loss_sum_batch += latent_loss.item()
            reconst_loss_sum_batch += reconst_loss.item()
            loss_sum_batch = loss_sum_batch + loss
            num_in_batch += 1

            if num_in_batch >= params.batch_size:
                mean_loss = loss_sum_batch / num_in_batch
                optim.zero_grad()
                mean_loss.backward()
                optim.step()

                print("(" + "|".join([
                    "%0.3f" % (latent_loss_sum_batch / num_in_batch),
                    "%0.3f" % (reconst_loss_sum_batch / num_in_batch)]) + ")",
                    end=' ', flush=True)

                latent_loss_sum_batch = 0.0
                reconst_loss_sum_batch = 0.0
                loss_sum_batch = 0.0
                num_in_batch = 0

            loss_sum += loss.item()
            loss_count += 1

        print('')
        loss_mean = loss_sum / loss_count

        metrics = [
            ('period', period),
            ('loss', round(loss_mean, 3))
        ]
        util.print_metrics(metrics)

        torch.save(
            latent_forger.state_dict(),
            'snapshots/' + params.header + LATENT_FORGER_FOOTER + '.pth')


def train_manipulators(params):
    speaker_categs = torch.load(params.speaker_categs_path)
    num_speakers, speaker_feature_dim = speaker_categs.size()

    describer_model = util.load_model(params.header + DESCRIBER_FOOTER)
    describer = Describer(describer_model, speaker_feature_dim)
    describer.load_state_dict(torch.load(
        'snapshots/' + params.header + DESCRIBER_FOOTER + '.pth'))
    describer.eval()

    reconstructor_model = util.load_model(params.header + RECONSTRUCTOR_FOOTER)
    reconstructor = Reconstructor(reconstructor_model, params.log_frac)
    reconstructor.load_state_dict(torch.load(
        'snapshots/' + params.header + RECONSTRUCTOR_FOOTER + '.pth'))

    latent_forger_model = util.load_model(params.header + LATENT_FORGER_FOOTER)
    latent_forger = LatentForger(latent_forger_model)
    latent_forger.load_state_dict(torch.load(
        'snapshots/' + params.header + LATENT_FORGER_FOOTER + '.pth'))
    latent_forger.train()

    examiner_model = util.load_model(params.header + EXAMINER_FOOTER)
    distinguisher_model = util.load_model(params.header + DISTINGUISHER_FOOTER)
    discriminator = PairDiscriminator(examiner_model, distinguisher_model)
    util.initialize(discriminator)
    discriminator.train()

    if example_tensor.is_cuda:
        speaker_categs = speaker_categs.cuda()
        describer = describer.cuda()
        reconstructor = reconstructor.cuda()
        latent_forger = latent_forger.cuda()
        discriminator = discriminator.cuda()

    optim = torch.optim.Adam(
        latent_forger.parameters(),
        lr=params.lr)
    advers_optim = torch.optim.Adam(
        discriminator.parameters(),
        lr=params.advers_lr)

    data_loader = VCTKLoader(
        params.data_path, example_tensor, features='log')
    data_iterator = iter(data_loader)

    forgery_categ_loss_sum_batch = 0.0
    activity_loss_sum_batch = 0.0
    pretend_latent_loss_sum_batch = 0.0
    pretend_reconst_loss_sum_batch = 0.0
    gen_loss_batch = 0.0
    advers_loss_batch = 0.0

    loss_sum_batch = 0.0
    discrim_loss_sum_batch = 0.0
    num_in_batch = 0

    period = 0
    while period < params.num_periods:
        period += 1

        loss_sum = 0.0
        loss_count = 0

        print(util.COMMENT_HEADER, end='')

        for _ in range(params.period_size):
            orig, orig_speaker = next(data_iterator)
            target, target_speaker = next(data_iterator)

            orig_categ = speaker_categs[orig_speaker].unsqueeze(0)
            target_categ = speaker_categs[target_speaker].unsqueeze(0)

            target_latent, metadata = describer.latent(target)
            target_reconst = reconstructor.reconst(target_latent, metadata)
            target_reconst = target_reconst.detach()

            orig_latent, metadata = describer.latent(orig)
            orig_latent = orig_latent.detach()

            forgery_latent_raw = latent_forger.modify_latent(
                orig_latent, orig_categ, target_categ)
            forgery = reconstructor.reconst(forgery_latent_raw, metadata)

            (forgery_latent, metadata, pred_forgery_categ) = \
                describer.describe(forgery)

            activity_orig = torch.exp(orig).mean(dim=1)
            activity_forgery = torch.exp(forgery).mean(dim=1)

            pretend_latent = latent_forger.modify_latent(
                forgery_latent, target_categ, orig_categ)
            pretend_reconst = reconstructor.reconst(pretend_latent, metadata)

            truth = 0 if (np.random.random() < 0.5) else 1

            if truth == 0:
                decision = discriminator.discriminate(
                    target_categ, target_reconst, forgery)
            else:
                decision = discriminator.discriminate(
                    target_categ, forgery, target_reconst)

            forgery_categ_loss = describer.categ_loss(
                pred_forgery_categ, target_categ)
            activity_loss = ((activity_orig - activity_forgery) ** 2).mean(
                dim=list(range(activity_orig.dim())))
            pretend_latent_loss = describer.latent_loss(
                pretend_latent, orig_latent)
            pretend_reconst_loss = reconstructor.reconst_loss(
                pretend_reconst, orig)
            gen_loss = discriminator.gen_loss(decision, truth)
            advers_loss = discriminator.advers_loss(decision, truth)

            loss = (params.categ_term * forgery_categ_loss +
                    params.activity_term * activity_loss +
                    pretend_latent_loss +
                    pretend_reconst_loss +
                    params.advers_term * gen_loss)
            discrim_loss = advers_loss

            forgery_categ_loss_sum_batch += forgery_categ_loss.item()
            activity_loss_sum_batch += activity_loss.item()
            pretend_latent_loss_sum_batch += pretend_latent_loss.item()
            pretend_reconst_loss_sum_batch += pretend_reconst_loss.item()
            gen_loss_batch += gen_loss.item()
            advers_loss_batch += advers_loss.item()

            loss_sum_batch = loss_sum_batch + loss
            discrim_loss_sum_batch = discrim_loss_sum_batch + discrim_loss
            num_in_batch += 1

            if num_in_batch >= params.batch_size:
                mean_discrim_loss = discrim_loss_sum_batch / num_in_batch
                if gen_loss_batch / num_in_batch <= 10.0:
                    advers_optim.zero_grad()
                    mean_discrim_loss.backward(retain_graph=True)
                    advers_optim.step()

                mean_loss = loss_sum_batch / num_in_batch
                optim.zero_grad()
                mean_loss.backward()
                if period >= 1:
                    optim.step()

                print("(" + "|".join([
                    "%0.3f" % (forgery_categ_loss_sum_batch / num_in_batch),
                    "%0.3f" % (activity_loss_sum_batch / num_in_batch),
                    "%0.3f" % (pretend_latent_loss_sum_batch / num_in_batch),
                    "%0.3f" % (pretend_reconst_loss_sum_batch / num_in_batch),
                    "%0.3f" % (gen_loss_batch / num_in_batch),
                    "%0.3f" % (advers_loss_batch / num_in_batch)
                ]) + ")", end=' ', flush=True)

                forgery_categ_loss_sum_batch = 0.0
                activity_loss_sum_batch = 0.0
                pretend_latent_loss_sum_batch = 0.0
                pretend_reconst_loss_sum_batch = 0.0
                gen_loss_batch = 0.0
                advers_loss_batch = 0.0

                loss_sum_batch = 0.0
                discrim_loss_sum_batch = 0.0
                num_in_batch = 0

            loss_sum += loss.item()
            loss_count += 1

        print('')
        loss_mean = loss_sum / loss_count

        metrics = [
            ('period', period),
            ('loss', round(loss_mean, 3))
        ]
        util.print_metrics(metrics)

        torch.save(
            latent_forger.state_dict(),
            'snapshots/' + params.header + LATENT_FORGER_FOOTER + '.pth')


def playground(params):
    speaker_categs = torch.load(params.speaker_categs_path)
    num_speakers, speaker_feature_dim = speaker_categs.size()

    describer_model = util.load_model(params.header + DESCRIBER_FOOTER)
    describer = Describer(
        describer_model, speaker_feature_dim)
    describer.eval()

    reconstructor_model = util.load_model(params.header + RECONSTRUCTOR_FOOTER)
    reconstructor = Reconstructor(reconstructor_model, params.log_frac)

    latent_forger_model = util.load_model(params.header + LATENT_FORGER_FOOTER)
    latent_forger = LatentForger(latent_forger_model)

    describer.load_state_dict(torch.load(
        'snapshots/' + params.header + DESCRIBER_FOOTER + '.pth',
        map_location=lambda storage, loc: storage))
    reconstructor.load_state_dict(torch.load(
        'snapshots/' + params.header + RECONSTRUCTOR_FOOTER + '.pth',
        map_location=lambda storage, loc: storage))
    latent_forger.load_state_dict(torch.load(
        'snapshots/' + params.header + LATENT_FORGER_FOOTER + '.pth',
        map_location=lambda storage, loc: storage))

    IPython.embed()


def main():
    parsed_args = parser.parse_args()
    params = Parameters()
    util.write_parsed_args(params, parsed_args)

    if params.rand_seed != -1:
        random.seed(params.rand_seed)
        np.random.seed(params.rand_seed)
        torch.random.manual_seed(params.rand_seed)

    if params.stage == "train_analysts":
        train_analysts(params)
    elif params.stage == "pretrain_manipulators":
        pretrain_manipulators(params)
    elif params.stage == "train_manipulators":
        train_manipulators(params)
    elif params.stage == "playground":
        playground(params)
    else:
        print("Unrecognized stage: " + params.stage)
        exit()


if __name__ == '__main__':
    main()
