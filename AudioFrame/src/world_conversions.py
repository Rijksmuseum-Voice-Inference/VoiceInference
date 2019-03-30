import numpy as np
import librosa
from matplotlib import pyplot as plt
import pyworld
from scipy.fftpack import dct, idct

EPSILON = 1e-8


def load_wav(file_name, sample_rate=None):
    return librosa.load(file_name, sr=sample_rate)[0]


def save_wav(file_name, sample_rate, samples):
    librosa.output.write_wav(file_name, samples, sample_rate)


def display_frames(frames):
    x, y = np.mgrid[:frames.shape[0], :(frames.shape[1] + 1)]
    plt.clf()
    fig = plt.figure(figsize=(8, 3))
    plt.pcolormesh(x, y, frames)
    plt.savefig('temp.png', dpi=800)
    plt.close(fig)


def ConversionSettings(
        sample_rate=16000,
        frame_period=5.0,
        coded_dim=36,
        f0_floor=71.0,
        f0_ceil=800.0):
    return {
        'sample_rate': sample_rate,
        'frame_period': frame_period,
        'coded_dim': coded_dim,
        'f0_floor': f0_floor,
        'f0_ceil': f0_ceil,
    }


def world_decompose(samples, settings):
    f0, timeaxis = pyworld.harvest(
        samples, settings['sample_rate'],
        frame_period=settings['frame_period'],
        f0_floor=settings['f0_floor'],
        f0_ceil=settings['f0_ceil'])
    spectral_env = pyworld.cheaptrick(
        samples, f0, timeaxis, settings['sample_rate'])
    aperiodicity = pyworld.d4c(
        samples, f0, timeaxis, settings['sample_rate'])
    return f0, timeaxis, spectral_env, aperiodicity


def world_encode_spectral_env(spectral_env, settings):
    mfcc = pyworld.code_spectral_envelope(
        spectral_env, settings['sample_rate'], settings['coded_dim'])
    return idct(mfcc) / np.sqrt(settings['coded_dim'] * 2)


def world_decode_spectral_env(spectral_env_mel, settings):
    mfcc = dct(spectral_env_mel) / np.sqrt(settings['coded_dim'] * 2)
    fftlen = pyworld.get_cheaptrick_fft_size(settings['sample_rate'])
    spectral_env = pyworld.decode_spectral_envelope(
        mfcc, settings['sample_rate'], fftlen)
    return spectral_env


def encode(samples, settings=None):
    if settings is None:
        settings = ConversionSettings()

    samples = samples.astype(np.float64)
    f0, timeaxis, spectral_env, aperiodicity = \
        world_decompose(samples, settings)
    spectral_env_mel = world_encode_spectral_env(spectral_env, settings)

    f0_norm = (f0 - settings['f0_floor']) / (
        settings['f0_ceil'] - settings['f0_floor'])
    f0_norm[f0_norm < 0.0] = 0.0
    f0_norm = f0_norm[:, np.newaxis]

    activation = (aperiodicity.mean(axis=1) < (1 - EPSILON))[:, np.newaxis]

    aperiodicity_mean = aperiodicity[
        activation[:, 0] == 1, :].mean(axis=0, keepdims=True)
    aperiodicity[activation[:, 0] == 0, :] += aperiodicity_mean - 1

    features = np.concatenate([spectral_env_mel, f0_norm], axis=1)

    return (features.astype(np.float32), activation, aperiodicity)


def decode(features, activation, aperiodicity, settings=None):
    if settings is None:
        settings = ConversionSettings()

    features = features.astype(np.float64)
    spectral_env_mel = features[:, :-1]
    f0 = settings['f0_floor'] + features[:, -1] * (
        settings['f0_ceil'] - settings['f0_floor'])

    f0[activation[:, 0] == 0] = 0.0
    aperiodicity[activation[:, 0] == 0] = 1.0

    spectral_env = world_decode_spectral_env(spectral_env_mel, settings)

    samples = pyworld.synthesize(
        f0, spectral_env, aperiodicity,
        settings['sample_rate'], settings['frame_period'])

    return samples.astype(np.float32), spectral_env, f0
