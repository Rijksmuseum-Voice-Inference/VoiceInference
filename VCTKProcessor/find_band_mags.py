import pickle
import numpy as np
import conversions

NUM_SPEAKERS = 100
SPEAKER_START_INDEX = 0
BAND_MAGS_PATH = "data/band_mags.npy"

with open("data/conv_options.pkl", 'rb') as f:
    conv_options = pickle.load(f)

window_power = conversions.get_window_power()

sum_band_mags = np.zeros((1, conv_options.fft_size // 2 + 1))
count_band_mags = np.zeros((1, conv_options.fft_size // 2 + 1))

for speaker in range(SPEAKER_START_INDEX, NUM_SPEAKERS):
    path = "data/speech_" + str(speaker) + ".npy"
    speech = np.load(path)

    path = "data/sizes_" + str(speaker) + ".npy"
    sizes = np.load(path)

    num_utterances = sizes.shape[0]
    indices = np.concatenate([[0], np.cumsum(sizes)])

    for utterance in range(num_utterances):
        start_index = indices[utterance]
        end_index = indices[utterance + 1]

        value = speech[start_index:end_index]
        energy = (value ** 2).sum(axis=1, keepdims=True)
        energy = energy * 2 - (value[:, 0:1] ** 2)
        energy = energy / window_power
        activation_mask = (np.sqrt(energy) > conv_options.noise_thres)

        sum_band_mags += np.sum(value * activation_mask, axis=0, keepdims=True)
        count_band_mags += np.sum(activation_mask, axis=0, keepdims=True)

count_band_mags = np.maximum(count_band_mags, 1)
band_mags = sum_band_mags / count_band_mags

np.save('data/band_mags.npy')
