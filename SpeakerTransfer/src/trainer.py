#!/usr/bin/env python3

import pdb
import random
import numpy as np
import torch
import util

from describer import Describer
from reconstructor import Reconstructor
from latent_forger import LatentForger
from discriminator import Discriminator
from loader import VCTKLoader


DESCRIBER_FOOTER = "_describer"
RECONSTRUCTOR_FOOTER = "_reconstructor"
LATENT_FORGER_FOOTER = "_latent_forger"
DISCRIMINATOR_FOOTER = "_discriminator"

example_tensor = torch.tensor(0.0)
if torch.cuda.is_available():
    example_tensor = example_tensor.cuda()


class Parameters:
    def __init__(self):
        self.data_path = "../VCTKProcessor/data/"
        self.speaker_categs_path = "../SpeakerFeatures/data/centers.pth"
        self.stage = ""
        self.header = "speaker_transfer"
        self.lr = 0.00004
        self.advers_lr = 0.00002
        self.categ_term = 0.02
        self.advers_term = 0.01
        self.batch_size = 32
        self.num_periods = 0
        self.period_size = 10000
        self.rand_seed = -1


parser = util.make_parser(Parameters(), "Latent Forger Model")


def train_analysts(params):
    speaker_categs = torch.load(params.speaker_categs_path)
    speaker_feature_dim = speaker_categs.size()[1]

    describer_model = util.load_model(params.header + DESCRIBER_FOOTER)
    describer = Describer(describer_model, speaker_feature_dim)
    util.initialize(describer)
    describer.train()

    reconstructor_model = util.load_model(params.header + RECONSTRUCTOR_FOOTER)
    reconstructor = Reconstructor(reconstructor_model)
    util.initialize(reconstructor)
    reconstructor.train()

    discriminator_model = util.load_model(params.header + DISCRIMINATOR_FOOTER)
    discriminator = Discriminator(discriminator_model)
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

    data_loader = VCTKLoader(params.data_path, example_tensor)
    data_iterator = iter(data_loader)

    categ_loss_sum_batch = 0.0
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

            (latent, metadata, pred_categ) = describer.describe(orig)
            reconst = reconstructor.reconst(latent, metadata)

            real_decision = discriminator.discriminate(orig, orig_categ)
            fake_decision = discriminator.discriminate(reconst, orig_categ)

            categ_loss = describer.categ_loss(pred_categ, orig_categ)
            reconst_loss = reconstructor.reconst_loss(reconst, orig)
            gen_loss = discriminator.gen_loss(fake_decision)
            advers_loss = discriminator.advers_loss(
                real_decision, fake_decision)

            loss = (
                params.categ_term * categ_loss +
                reconst_loss +
                params.advers_term * gen_loss)
            discrim_loss = advers_loss

            categ_loss_sum_batch += categ_loss.item()
            reconst_loss_sum_batch += reconst_loss.item()
            gen_loss_batch += gen_loss.item()
            advers_loss_batch += advers_loss.item()

            loss_sum_batch = loss_sum_batch + loss
            discrim_loss_sum_batch = discrim_loss_sum_batch + discrim_loss
            num_in_batch += 1

            if num_in_batch >= params.batch_size:
                mean_discrim_loss = discrim_loss_sum_batch / num_in_batch
                advers_optim.zero_grad()
                mean_discrim_loss.backward(retain_graph=True)
                advers_optim.step()

                mean_loss = loss_sum_batch / num_in_batch
                optim.zero_grad()
                mean_loss.backward()
                optim.step()

                print("(" + "|".join([
                    "%0.3f" % (categ_loss_sum_batch / num_in_batch),
                    "%0.3f" % (reconst_loss_sum_batch / num_in_batch),
                    "%0.3f" % (gen_loss_batch / num_in_batch),
                    "%0.3f" % (advers_loss_batch / num_in_batch)]) + ")",
                    end=' ', flush=True)

                categ_loss_sum_batch = 0.0
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
    describer = Describer(
        describer_model, speaker_feature_dim)
    describer.load_state_dict(torch.load(
        'snapshots/' + params.header + DESCRIBER_FOOTER + '.pth'))
    describer.eval()

    reconstructor_model = util.load_model(params.header + RECONSTRUCTOR_FOOTER)
    reconstructor = Reconstructor(reconstructor_model)
    reconstructor.load_state_dict(torch.load(
        'snapshots/' + params.header + RECONSTRUCTOR_FOOTER + '.pth'))
    reconstructor.eval()

    latent_forger_model = util.load_model(params.header + LATENT_FORGER_FOOTER)
    latent_forger = LatentForger(latent_forger_model)
    util.initialize(latent_forger)
    latent_forger.train()

    if example_tensor.is_cuda:
        speaker_categs = speaker_categs.cuda()
        describer = describer.cuda()
        reconstructor = reconstructor.cuda()
        latent_forger = latent_forger.cuda()

    optim = torch.optim.Adam(latent_forger.parameters(), lr=params.lr)

    data_loader = VCTKLoader(params.data_path, example_tensor)
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
    describer = Describer(
        describer_model, speaker_feature_dim)
    describer.load_state_dict(torch.load(
        'snapshots/' + params.header + DESCRIBER_FOOTER + '.pth'))
    describer.eval()

    reconstructor_model = util.load_model(params.header + RECONSTRUCTOR_FOOTER)
    reconstructor = Reconstructor(reconstructor_model)
    reconstructor.load_state_dict(torch.load(
        'snapshots/' + params.header + RECONSTRUCTOR_FOOTER + '.pth'))
    reconstructor.eval()

    latent_forger_model = util.load_model(params.header + LATENT_FORGER_FOOTER)
    latent_forger = LatentForger(latent_forger_model)
    latent_forger.load_state_dict(torch.load(
        'snapshots/' + params.header + LATENT_FORGER_FOOTER + '.pth'))
    latent_forger.train()

    discriminator_model = util.load_model(params.header + DISCRIMINATOR_FOOTER)
    discriminator = Discriminator(discriminator_model)
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

    data_loader = VCTKLoader(params.data_path, example_tensor)
    data_iterator = iter(data_loader)

    forgery_categ_loss_sum_batch = 0.0
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
            orig_categ = speaker_categs[orig_speaker].unsqueeze(0)

            forgery_categ = speaker_categs[
                np.random.randint(num_speakers)].unsqueeze(0)

            orig_latent, metadata = describer.latent(orig)
            orig_latent = orig_latent.detach()
            orig_reconst = reconstructor.reconst(orig_latent, metadata)
            orig_reconst = orig_reconst.detach()
            forgery_latent_raw = latent_forger.modify_latent(
                orig_latent, orig_categ, forgery_categ)
            forgery = reconstructor.reconst(forgery_latent_raw, metadata)
            (forgery_latent, metadata, pred_forgery_categ) = \
                describer.describe(forgery)

            pretend_latent = latent_forger.modify_latent(
                forgery_latent, forgery_categ, orig_categ)
            pretend_reconst = reconstructor.reconst(pretend_latent, metadata)

            real_decision = discriminator.discriminate(
                orig_reconst, orig_categ)
            fake_decision = discriminator.discriminate(
                forgery, forgery_categ)

            forgery_categ_loss = describer.categ_loss(
                pred_forgery_categ, forgery_categ)
            pretend_latent_loss = describer.latent_loss(
                pretend_latent, orig_latent)
            pretend_reconst_loss = reconstructor.reconst_loss(
                pretend_reconst, orig)
            gen_loss = discriminator.gen_loss(fake_decision)
            advers_loss = discriminator.advers_loss(
                real_decision, fake_decision)

            loss = (params.categ_term * forgery_categ_loss +
                    pretend_latent_loss +
                    pretend_reconst_loss +
                    params.advers_term * gen_loss)
            discrim_loss = advers_loss

            forgery_categ_loss_sum_batch += forgery_categ_loss.item()
            pretend_latent_loss_sum_batch += pretend_latent_loss.item()
            pretend_reconst_loss_sum_batch += pretend_reconst_loss.item()
            gen_loss_batch += gen_loss.item()
            advers_loss_batch += advers_loss.item()

            loss_sum_batch = loss_sum_batch + loss
            discrim_loss_sum_batch = discrim_loss_sum_batch + discrim_loss
            num_in_batch += 1

            if num_in_batch >= params.batch_size:
                mean_discrim_loss = discrim_loss_sum_batch / num_in_batch
                advers_optim.zero_grad()
                mean_discrim_loss.backward(retain_graph=True)
                advers_optim.step()

                mean_loss = loss_sum_batch / num_in_batch
                optim.zero_grad()
                mean_loss.backward()
                optim.step()

                print("(" + "|".join([
                    "%0.3f" % (forgery_categ_loss_sum_batch / num_in_batch),
                    "%0.3f" % (pretend_latent_loss_sum_batch / num_in_batch),
                    "%0.3f" % (pretend_reconst_loss_sum_batch / num_in_batch),
                    "%0.3f" % (gen_loss_batch / num_in_batch),
                    "%0.3f" % (advers_loss_batch / num_in_batch)
                ]) + ")", end=' ', flush=True)

                forgery_categ_loss_sum_batch = 0.0
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
    describer.load_state_dict(torch.load(
        'snapshots/' + params.header + DESCRIBER_FOOTER + '.pth'))
    describer.eval()

    reconstructor_model = util.load_model(params.header + RECONSTRUCTOR_FOOTER)
    reconstructor = Reconstructor(reconstructor_model)
    reconstructor.load_state_dict(torch.load(
        'snapshots/' + params.header + RECONSTRUCTOR_FOOTER + '.pth'))
    reconstructor.eval()

    latent_forger_model = util.load_model(params.header + LATENT_FORGER_FOOTER)
    latent_forger = LatentForger(latent_forger_model)
    latent_forger.load_state_dict(torch.load(
        'snapshots/' + params.header + LATENT_FORGER_FOOTER + '.pth'))
    latent_forger.train()

    if example_tensor.is_cuda:
        speaker_categs = speaker_categs.cuda()
        describer = describer.cuda()
        reconstructor = reconstructor.cuda()
        latent_forger = latent_forger.cuda()

    try:
        describer.load_state_dict(torch.load(
            'snapshots/' + params.header + DESCRIBER_FOOTER + '.pth'))
        reconstructor.load_state_dict(torch.load(
            'snapshots/' + params.header + RECONSTRUCTOR_FOOTER + '.pth'))
        latent_forger.load_state_dict(torch.load(
            'snapshots/' + params.header + LATENT_FORGER_FOOTER + '.pth'))
    except Exception:
        print("Couldn't load all snapshots!")
        pass

    pdb.set_trace()


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
