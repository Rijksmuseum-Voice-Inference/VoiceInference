#!/usr/bin/env python3

import os
import pickle
import numpy as np
import conversions

VCTK_PATH = "../VCTK-Corpus/wav48/"

conv_options = conversions.AudioFrameConvOptions()
sample_rate = 48000

with open('data/conv_options.pkl', 'wb') as f:
    pickle.dump(conv_options, f)

file_id = 0
metadata = []

for speaker in os.listdir(VCTK_PATH):
    speaker_path = os.path.join(VCTK_PATH, speaker)

    speech_path = os.path.join('data', 'speech_' + str(file_id) + '.npy')
    sizes_path = os.path.join('data', 'sizes_' + str(file_id) + '.npy')
    file_id += 1

    utterances = []
    speech = []
    sizes = []

    for utterance in os.listdir(speaker_path):
        if utterance.find('.wav') == -1:
            continue

        utterance_path = os.path.join(speaker_path, utterance)
        _, samples = conversions.load_wav(utterance_path)
        frames, _ = conversions.encode(sample_rate, samples, conv_options)

        utterances.append(utterance)
        speech.append(frames)
        sizes.append(frames.shape[0])

    metadata.append((speaker, utterances))
    speech = np.concatenate(speech, axis=0)
    sizes = np.array(sizes)
    np.save(speech_path, speech)
    np.save(sizes_path, sizes)

with open('data/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
