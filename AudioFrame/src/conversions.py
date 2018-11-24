import numpy as np
import librosa
from matplotlib import pyplot as plt

EPSILON = 1e-8


def load_wav(file_name, sample_rate=None):
    return librosa.load(file_name, sr=sample_rate)[0]


def save_wav(file_name, sample_rate, samples):
    librosa.output.write_wav(file_name, samples, sample_rate)


def display_frames(frames):
    x, y = np.mgrid[:frames.shape[0], :frames.shape[1] - 1]
    plt.pcolormesh(x, y, frames[:, :-1])
    plt.show(block=False)


class AudioFrameConvOptions:
    def __init__(self):
        self.window_length = 0.025
        self.interval_length = 0.005
        self.fft_size = 512
        self.noise_thres = 0.001


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

    energy = (frames ** 2).sum(axis=1, keepdims=True)
    energy = np.sqrt((energy * 2 - frames[:, 0:1] ** 2) / options.fft_size)
    frames = np.concatenate([frames, energy], axis=1)

    noise_thres_f = np.sqrt((options.noise_thres ** 2) / options.fft_size)
    frames = np.log(1.0 + (frames / noise_thres_f))
    frames = frames + np.log(noise_thres_f)

    return frames, phase_data


def decode(sample_rate, frames, num_iters,
           options=default_options, phase_data=None):
    window = round(sample_rate * options.window_length)
    interval = round(sample_rate * options.interval_length)

    noise_thres_f = np.sqrt((options.noise_thres ** 2) / options.fft_size)
    frames = np.maximum(frames[:, :-1] - np.log(noise_thres_f), 0)
    frames = (np.exp(frames) - 1.0) * noise_thres_f

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


def fourier_time_series(samples, window, interval, fft_size):
    num_intervals = get_num_intervals(samples.shape[0], window, interval)
    window_indices = get_window_indices(num_intervals, window, interval)

    time_slices = np.pad(
        samples[window_indices], [(0, 0), (0, fft_size - window)], 'constant')
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
