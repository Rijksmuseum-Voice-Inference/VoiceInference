import torch
import torch.nn.functional as F
import numpy as np


def get_window_fn(window):
    window_fn = np.cos(2 * np.pi * np.arange(window) / window)
    window_fn = 25 / 46 - 21 / 46 * window_fn
    return window_fn[np.newaxis, :]


class STFT(torch.nn.Module):
    def __init__(self, segment_length=400, filter_length=512,
                 hop_length=80, eps=1e-8):
        super(STFT, self).__init__()

        self.segment_length = segment_length
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.eps = eps

        self.forward_transform = None
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]),
             np.imag(fourier_basis[:cutoff, :])])
        fourier_basis[:, self.segment_length:] = 0
        fourier_basis[:, :self.segment_length] *= get_window_fn(segment_length)
        fourier_basis /= np.sqrt(filter_length * segment_length)

        forward_basis = torch.FloatTensor(
            fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(fourier_basis).T[:, None, :])

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        input_data = input_data.view(num_batches, 1, num_samples)
        pad_amount = (
            self.hop_length + self.filter_length - self.segment_length)
        input_data = F.pad(input_data, (0, pad_amount))

        forward_transform = F.conv1d(
            input_data,
            self.forward_basis,
            stride=self.hop_length)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2).clamp(min=self.eps)
        phase = torch.atan2(imag_part, real_part)

        return magnitude, phase

    def inverse(self, magnitude, phase, num_samples=None):
        if num_samples is None:
            num_samples = (
                (magnitude.shape[2] - 1) * self.hop_length +
                self.segment_length)

        recombine_magnitude_phase = torch.cat(
            [magnitude * torch.cos(phase),
             magnitude * torch.sin(phase)], dim=1)

        weights_kernel = torch.ones_like(self.inverse_basis[:1, :, :])
        weights_kernel[:, :, self.segment_length:] = 0
        weights = F.conv_transpose1d(
            torch.ones_like(recombine_magnitude_phase[:, :1, :]),
            weights_kernel,
            stride=self.hop_length)
        weights[weights <= 1] = 1

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            self.inverse_basis,
            stride=self.hop_length) / weights

        inverse_transform = inverse_transform[:, 0, :num_samples]
        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction
