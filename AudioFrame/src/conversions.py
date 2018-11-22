import warnings
import numpy as np
from scipy.io import wavfile
from scipy.special import lambertw
from matplotlib import pyplot as plt

EPSILON = 1e-8


def load_wav(file_name):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        sample_rate, samples = wavfile.read(file_name)

    if samples.dtype == np.int16:
        samples = np.maximum(samples, -2 ** 15 + 1) / (2 ** 15 - 1)
    elif samples.dtype == np.float:
        pass
    else:
        raise RuntimeError("Unsupported data type: " + str(samples.dtype))

    if samples.ndim == 2:
        samples = (samples[:, 0] + samples[:, 1]) / 2.0

    return sample_rate, samples


def save_wav(file_name, sample_rate, samples):
    if samples.dtype == np.int16:
        pass
    elif samples.dtype == np.float:
        samples = np.clip(
            np.round(samples * (2 ** 15)),
            -2 ** 15, 2 ** 15 - 1).astype(np.int16)
    else:
        raise RuntimeError("Unsupported data type: " + str(samples.dtype))

    wavfile.write(file_name, sample_rate, samples)


def display_frames(frames):
    x, y = np.mgrid[:frames.shape[0], :frames.shape[1] - 1]
    plt.pcolormesh(x, y, frames[:, :-1])
    plt.show(block=False)


class AudioFrameConvOptions:
    def __init__(self):
        self.window_length = 0.05
        self.interval_length = 0.004
        self.crop_frac = 1.0
        self.num_buckets = 300


default_options = AudioFrameConvOptions()


def encode(sample_rate, samples, options=default_options):
    length = samples.shape[0]
    window = round(sample_rate * options.window_length)
    interval = round(sample_rate * options.interval_length)

    expanded_length = expand_to_fit(length, window, interval)
    samples = np.pad(samples, (0, expanded_length - length), 'constant')

    raw_frames_cmp = fourier_time_series(samples, window, interval)
    raw_frames = np.abs(raw_frames_cmp)
    phase_data = np.angle(raw_frames_cmp)

    raw_frame_size = window // 2 + 1
    thresholds = bucket_thresholds(
        options.num_buckets,
        int(options.crop_frac * raw_frame_size),
        raw_frame_size)

    frames = np.maximum(bucketize(raw_frames, thresholds), EPSILON)
    energy = np.sqrt((frames ** 2).mean(axis=1, keepdims=True))
    frames = np.concatenate([frames, energy], axis=1)
    frames = np.log(1.0 + frames)

    return frames, phase_data


def decode(sample_rate, frames, num_iters,
           options=default_options, phase_data=None):
    window = round(sample_rate * options.window_length)
    interval = round(sample_rate * options.interval_length)

    raw_frame_size = window // 2 + 1
    thresholds = bucket_thresholds(
        options.num_buckets,
        int(options.crop_frac * raw_frame_size),
        raw_frame_size)
    raw_frames = unbucketize(np.exp(frames[:, :-1]) - 1.0, thresholds)

    if phase_data is None:
        phase_data = np.random.uniform(
            -np.pi / 2, np.pi / 2, size=raw_frames.shape)
        phase_data = np.cumsum(phase_data, axis=0)
        phase_data = phase_data + np.random.uniform(
            -np.pi, np.pi,
            size=(1, raw_frames.shape[1]))

    raw_frames_iter = raw_frames * np.exp(1.0j * phase_data)

    for i in range(1, num_iters):
        buckets = inv_fourier_time_series(raw_frames_iter, window, interval)
        raw_frames_iter = fourier_time_series(buckets, window, interval)
        raw_frames_iter = raw_frames * np.exp(1.0j * np.angle(raw_frames_iter))

    return inv_fourier_time_series(raw_frames_iter, window, interval)


def bucket_thresholds(num_buckets, max_threshold, full_length):
    if num_buckets > max_threshold:
        raise RuntimeError(
            "Can't make %d buckets from %d" % (num_buckets, max_threshold))
    if num_buckets == max_threshold:
        return np.concatenate([np.arange(max_threshold), [full_length]])

    ratio = num_buckets / max_threshold
    inner_term = -np.exp(-ratio) * ratio
    lambert_term = np.real(lambertw(inner_term, -1))
    numerator = -max_threshold * lambert_term - num_buckets
    scale = numerator / (num_buckets * max_threshold)
    thresholds = 1.0 / scale * (np.exp(scale * np.arange(1, num_buckets)) - 1)
    thresholds = np.concatenate([[0], thresholds, [full_length]])
    return thresholds.astype(np.int)


def bucketize(raw_frames, thresholds):
    sections = np.split(raw_frames, thresholds[1:-1], axis=1)
    means = [np.mean(section, axis=1, keepdims=True) for section in sections]
    return np.concatenate(means, axis=1)


def unbucketize(frames, thresholds):
    counts = np.diff(thresholds)
    return np.repeat(frames, counts, axis=1)


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


def fourier_time_series(samples, window, interval):
    num_intervals = get_num_intervals(samples.shape[0], window, interval)
    window_indices = get_window_indices(num_intervals, window, interval)

    time_slices = samples[window_indices]
    fourier_time_slices = np.zeros(
        (num_intervals, window // 2 + 1), dtype='c8')

    for i in range(num_intervals):
        fourier_time_slices[i] = np.fft.rfft(time_slices[i])

    return fourier_time_slices


def inv_fourier_time_series(fourier_time_slices, window, interval):
    num_intervals = fourier_time_slices.shape[0]
    length = (num_intervals - 1) * interval + window
    window_indices = get_window_indices(num_intervals, window, interval)

    time_slices = np.zeros((num_intervals, window))
    window_weights = np.zeros(length)
    samples = np.zeros(length)

    for i in range(num_intervals):
        time_slices[i] = np.fft.irfft(fourier_time_slices[i], window)

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
