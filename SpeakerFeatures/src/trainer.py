import random
import numpy as np
import torch
import util
import loader

ENCODER_FOOTER = "_encoder"
DISCRIMINATOR_FOOTER = "_discriminator"

example_tensor = torch.tensor(0.0)
if torch.cuda.is_available():
    example_tensor = example_tensor.cuda()


class Parameters:
    def __init__(self):
        self.data_path = "../VCTKProcessor/data/"
        self.header = "speaker_features"
        self.speaker_take_count = 15
        self.utterance_take_count = 20
        self.lr = 0.0001
        self.batch_size = 64
        self.average_count = 100
        self.cluster_term = 0.05
        self.num_periods = 0
        self.period_size = 10000
        self.rand_seed = -1


parser = util.make_parser(Parameters(), "Speaker Feature Extractor")


def train(params):
    encoder = util.load_model(params.header + ENCODER_FOOTER)
    util.initialize(encoder)
    encoder.train()

    discriminator = util.load_model(params.header + DISCRIMINATOR_FOOTER)
    util.initialize(discriminator)
    discriminator.train()

    if example_tensor.is_cuda:
        encoder = encoder.cuda()
        discriminator = discriminator.cuda()

    optim = torch.optim.Adam(
        torch.nn.Sequential(encoder, discriminator).parameters(),
        lr=params.lr)
    mse = torch.nn.MSELoss()
    bce = torch.nn.BCEWithLogitsLoss()

    speaker_averages = {}
    average_decay = 1.0 - 1.0 / params.average_count
    data_loader = loader.VCTKLoader(
        params.data_path, example_tensor,
        speaker_take_count=params.speaker_take_count,
        utterance_take_count=params.utterance_take_count)

    def update_average(speaker, encoded):
        if speaker not in speaker_averages:
            speaker_averages[speaker] = encoded
        else:
            speaker_averages[speaker] = (
                average_decay * speaker_averages[speaker] +
                (1.0 - average_decay) * encoded
            )

    cluster_loss_sum_batch = 0.0
    discrim_loss_sum_batch = 0.0
    loss_sum_batch = 0.0
    num_in_batch = 0
    data_iterator = iter(data_loader)

    period = 0
    while period < params.num_periods:
        period += 1

        loss_sum_print = 0.0
        loss_count_print = 0

        print(util.COMMENT_HEADER, end='')

        for _ in range(params.period_size):
            utterance_1, speaker_1 = next(data_iterator)
            utterance_2, speaker_2 = next(data_iterator)

            encoded_1 = encoder(utterance_1)
            encoded_2 = encoder(utterance_2)

            update_average(speaker_1, encoded_1.detach())
            update_average(speaker_2, encoded_2.detach())

            discrim_1 = discriminator(
                torch.cat([encoded_1, encoded_1], dim=1))
            discrim_2 = discriminator(
                torch.cat([encoded_2, encoded_2], dim=1))
            discrim_3 = discriminator(
                torch.cat([encoded_1, encoded_2], dim=1))
            discrim_4 = discriminator(
                torch.cat([encoded_2, encoded_1], dim=1))

            compare = 1 if speaker_1 == speaker_2 else 0
            compare = example_tensor.new_tensor(compare).reshape(1, 1)
            same = example_tensor.new_tensor(1).reshape(1, 1)

            discrim_loss = (
                bce(discrim_1, same) +
                bce(discrim_2, same) +
                bce(discrim_3, compare) +
                bce(discrim_4, compare)) / 4.0

            cluster_loss = (
                mse(encoded_1, speaker_averages[speaker_1]) +
                mse(encoded_2, speaker_averages[speaker_2])) / 2.0

            loss = params.cluster_term * cluster_loss + discrim_loss

            cluster_loss_sum_batch += cluster_loss.item()
            discrim_loss_sum_batch += discrim_loss.item()
            loss_sum_batch = loss_sum_batch + loss
            num_in_batch += 1

            if num_in_batch >= params.batch_size:
                mean_loss = loss_sum_batch / num_in_batch
                optim.zero_grad()
                mean_loss.backward()
                optim.step()

                loss_sum_print += mean_loss.item()
                loss_count_print += 1

                print("(" + "|".join([
                    "%0.3f" % (cluster_loss_sum_batch / num_in_batch),
                    "%0.3f" % (discrim_loss_sum_batch / num_in_batch)]) + ")",
                    end=' ', flush=True)

                cluster_loss_sum_batch = 0.0
                discrim_loss_sum_batch = 0.0
                loss_sum_batch = 0.0
                num_in_batch = 0

        print('')
        loss_mean = loss_sum_print / loss_count_print

        metrics = [
            ('period', period),
            ('loss', round(loss_mean, 3))
        ]
        util.print_metrics(metrics)

        torch.save(
            encoder.state_dict(),
            'snapshots/' + params.header + ENCODER_FOOTER + '.pth')

        torch.save(
            discriminator.state_dict(),
            'snapshots/' + params.header + DISCRIMINATOR_FOOTER + '.pth')


def main():
    parsed_args = parser.parse_args()
    params = Parameters()
    util.write_parsed_args(params, parsed_args)

    if params.rand_seed != -1:
        random.seed(params.rand_seed)
        np.random.seed(params.rand_seed)
        torch.random.manual_seed(params.rand_seed)

    train(params)


if __name__ == '__main__':
    main()
