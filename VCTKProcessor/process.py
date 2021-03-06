#!/usr/bin/env python3

import os
from multiprocessing import Pool
import pickle
import numpy as np
import conversions

NUM_PROCESSES = 20
SAMPLE_RATE = 16000
VCTK_PATH = "../VCTK-Corpus/wav48/"

conv_options = conversions.AudioFrameConvOptions()
sample_rate = 48000

with open('data/conv_options.pkl', 'wb') as f:
    pickle.dump(conv_options, f, protocol=2)

file_id = 0
metadata = []


def task(file_id, speaker_path, utterances):
    speech_path = os.path.join('data', 'speech_' + str(file_id) + '.npy')
    phase_path = os.path.join('data', 'phase_' + str(file_id) + '.npy')
    sizes_path = os.path.join('data', 'sizes_' + str(file_id) + '.npy')

    speech = []
    phase = []
    sizes = []

    for utterance in utterances:
        utterance_path = os.path.join(speaker_path, utterance)
        samples = conversions.load_wav(utterance_path, SAMPLE_RATE)
        frames, pdata = conversions.encode(SAMPLE_RATE, samples, conv_options)

        speech.append(frames.astype(np.float32))
        phase.append(pdata.astype(np.float32))
        sizes.append(frames.shape[0])

    speech = np.concatenate(speech, axis=0)
    phase = np.concatenate(phase, axis=0)
    sizes = np.array(sizes)
    np.save(speech_path, speech)
    np.save(phase_path, phase)
    np.save(sizes_path, sizes)


pool = Pool(NUM_PROCESSES)

for speaker in sorted(os.listdir(VCTK_PATH)):
    speaker_path = os.path.join(VCTK_PATH, speaker)
    utterances = sorted(
        [p for p in os.listdir(speaker_path) if p.find('.wav') != -1])
    metadata.append((speaker, utterances))

    pool.apply_async(task, [file_id, speaker_path, utterances])
    file_id += 1

pool.close()
pool.join()

with open('data/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f, protocol=2)
