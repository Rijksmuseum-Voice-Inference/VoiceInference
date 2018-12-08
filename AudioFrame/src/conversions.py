import os
import numpy as np
import librosa
from matplotlib import pyplot as plt
import torch

EPSILON = 1e-8
mel_fbank = np.load(
    os.path.join(os.path.dirname(__file__), 'mel-fbank80.npy')).T


def load_wav(file_name, sample_rate=None):
    return librosa.load(file_name, sr=sample_rate)[0]


def save_wav(file_name, sample_rate, samples):
    librosa.output.write_wav(file_name, samples, sample_rate)


def display_frames(frames):
    x, y = np.mgrid[:frames.shape[0], :frames.shape[1]]
    plt.clf()
    plt.pcolormesh(x, y, frames)
    plt.savefig('temp.png', dpi=800)


class AudioFrameConvOptions:
    def __init__(self):
        self.window_length = 0.025
        self.interval_length = 0.005
        self.fft_size = 512
        self.noise_thres = 0.002
        self.typical_sig = 0.02
        self.outlier_percentile = 99.9


default_options = AudioFrameConvOptions()


def encode(sample_rate, samples, options=default_options):
    length = samples.shape[0]
    window = round(sample_rate * options.window_length)
    interval = round(sample_rate * options.interval_length)

    expanded_length = expand_to_fit(length, window, interval)
    samples = np.pad(samples, (0, expanded_length - length), 'constant')

    frames_cmp = fourier_time_series(
        samples, window, interval, options.fft_size)
    frames = np.abs(frames_cmp)
    phase_data = np.angle(frames_cmp)

    return frames, phase_data


def decode(sample_rate, frames, num_iters,
           options=default_options, phase_data=None):
    window = round(sample_rate * options.window_length)
    interval = round(sample_rate * options.interval_length)

    if phase_data is None:
        phase_data = np.random.uniform(
            -np.pi / 2, np.pi / 2, size=frames.shape)
        phase_data = np.cumsum(phase_data, axis=0)
        phase_data = phase_data + np.random.uniform(
            -np.pi, np.pi,
            size=(1, frames.shape[1]))

    frames_iter = frames * np.exp(1.0j * phase_data)

    for i in range(1, num_iters):
        buckets = inv_fourier_time_series(
            frames_iter, window, interval, options.fft_size)
        frames_iter = fourier_time_series(
            buckets, window, interval, options.fft_size)
        frames_iter = frames * np.exp(1.0j * np.angle(frames_iter))

    return inv_fourier_time_series(
        frames_iter, window, interval, options.fft_size)


def to_log(mag_frames, options=default_options):
    noise_thres_f = np.sqrt((options.noise_thres ** 2) / options.fft_size)
    typical_sig_f = np.sqrt((options.typical_sig ** 2) / options.fft_size)

    return (np.log(1.0 + mag_frames / noise_thres_f) +
            np.log(noise_thres_f) - np.log(typical_sig_f))


def to_backward_compatible(log_frames):
    noise_thres_f = np.sqrt(
        (default_options.noise_thres ** 2) / default_options.fft_size)
    typical_sig_f = np.sqrt(
        (default_options.typical_sig ** 2) / default_options.fft_size)

    log_frames_pos = log_frames + np.log(typical_sig_f) - np.log(noise_thres_f)
    mag_frames = torch.clamp(
        (torch.exp(log_frames_pos) - 1.0) * noise_thres_f, min=0.0)

    C = 4.41941738242e-05
    return torch.log(C + mag_frames / np.sqrt(get_window_power()))


def to_backward_compatible_with_energy(log_frames):
    noise_thres_f = np.sqrt(
        (default_options.noise_thres ** 2) / default_options.fft_size)
    typical_sig_f = np.sqrt(
        (default_options.typical_sig ** 2) / default_options.fft_size)

    log_frames_pos = log_frames + np.log(typical_sig_f) - np.log(noise_thres_f)
    mag_frames = torch.clamp(
        (torch.exp(log_frames_pos) - 1.0) * noise_thres_f, min=0.0)

    energy = (mag_frames ** 2).sum(dim=1, keepdim=True)
    energy = torch.sqrt(
        (energy * 2 - mag_frames[:, 0:1, :] ** 2) / default_options.fft_size)

    mag_frames = torch.cat([mag_frames, energy], dim=1)

    C = 4.41941738242e-05
    return torch.log(C + mag_frames / np.sqrt(get_window_power()))


def to_mag_norm(mag_frames, band_mags, options=default_options):
    target_mag = np.log(band_mags)
    target_mag = target_mag - target_mag.min() + 1
    target_mag = target_mag / target_mag.max()

    norm_frames = mag_frames / band_mags * target_mag

    outlier_mag = np.percentile(norm_frames, options.outlier_percentile)
    norm_frames[norm_frames > outlier_mag] = outlier_mag

    return norm_frames


def to_two(mag_frames, band_mags, options=default_options):
    return np.concatenate(
        [to_log(mag_frames, options),
         to_mag_norm(mag_frames, band_mags)], axis=1)


def to_mel(mag_frames):
    feats = mag_frames.dot(mel_fbank)
    feats = np.log(np.maximum(feats, 1e-14))
    feats = feats - feats.min()
    feats = feats / feats.max() * 0.999

    return feats


def from_log(log_frames, options=default_options):
    noise_thres_f = np.sqrt((options.noise_thres ** 2) / options.fft_size)
    typical_sig_f = np.sqrt((options.typical_sig ** 2) / options.fft_size)

    log_frames_pos = np.maximum(
        log_frames + np.log(typical_sig_f) - np.log(noise_thres_f), 0.0)

    return (np.exp(log_frames_pos) - 1.0) * noise_thres_f


def from_mag_norm(norm_frames, band_mags, options=default_options):
    target_mag = np.log(band_mags)
    target_mag = target_mag - target_mag.min() + 1
    target_mag = target_mag / target_mag.max()

    return np.maximum(norm_frames, 0) / target_mag * band_mags


def from_two(two_frames, band_mags, options=default_options):
    count = two_frames.shape[1] // 2
    norm_frames = two_frames[:, count:]
    return from_mag_norm(norm_frames, band_mags, options)


def expand_to_fit(length, window, interval):
    length_intervals = length - window
    num_intervals = get_num_intervals(length, window, interval)
    length_intervals = num_intervals * interval
    return length_intervals + window


def get_num_intervals(length, window, interval):
    return (length - window + 1 + interval - 1) // interval


def get_window_indices(num_intervals, window, interval):
    first_window_index = np.arange(window)
    interval_offsets = np.arange(num_intervals) * interval
    return np.tile(
        first_window_index, (num_intervals, 1)) + \
        interval_offsets[:, np.newaxis]


def get_window_fn(window):
    window_fn = np.cos(2 * np.pi * np.arange(window) / window)
    window_fn = 25 / 46 - 21 / 46 * window_fn
    return window_fn[np.newaxis, :]


def get_window_power():
    return 1691 / 4232


def fourier_time_series(samples, window, interval, fft_size):
    num_intervals = get_num_intervals(samples.shape[0], window, interval)
    window_indices = get_window_indices(num_intervals, window, interval)

    time_slices = samples[window_indices]
    time_slices = time_slices * get_window_fn(window)
    time_slices = np.pad(
        time_slices, [(0, 0), (0, fft_size - window)], 'constant')

    fourier_time_slices = np.zeros(
        (num_intervals, fft_size // 2 + 1), dtype='c8')

    for i in range(num_intervals):
        fourier_time_slices[i] = np.fft.rfft(time_slices[i])

    return fourier_time_slices / np.sqrt(fft_size * window)


def inv_fourier_time_series(fourier_time_slices, window, interval, fft_size):
    fourier_time_slices = fourier_time_slices * np.sqrt(fft_size * window)

    num_intervals = fourier_time_slices.shape[0]
    length = (num_intervals - 1) * interval + window
    window_indices = get_window_indices(num_intervals, window, interval)

    time_slices = np.zeros((num_intervals, window))
    window_weights = np.zeros(length)
    samples = np.zeros(length)

    for i in range(num_intervals):
        time_slices[i] = np.fft.irfft(
            fourier_time_slices[i], fft_size)[:window]

    time_slices = time_slices / get_window_fn(window)

    window_weight_profile = np.concatenate([
        np.linspace(0, 1, interval),
        np.ones(window - 2 * interval),
        np.linspace(1, 0, interval)])
    window_weight_profile[0] = EPSILON
    window_weight_profile[-1] = EPSILON

    for (i, window_index) in enumerate(window_indices):
        window_weights[window_index] += window_weight_profile
        samples[window_index] += window_weight_profile * time_slices[i]

    samples = samples / window_weights
    return samples
