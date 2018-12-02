#!/usr/bin/env python3

import os
import pickle
import numpy as np
import torch
import util
import conversions
from loader import to_torch, from_torch

from describer import Describer
from reconstructor import Reconstructor

NUM_SPEAKERS = 100
SPEAKER_START_INDEX = 0
CATEG_SIZE = 128
DESCRIBER_NAME = "speaker_transfer_describer"
DESCRIBER_SNAPSHOT_PATH = "snapshots/speaker_transfer_describer.pth"
RECONSTRUCTOR_NAME = "speaker_transfer_reconstructor"
RECONSTRUCTOR_SNAPSHOT_PATH = "snapshots/speaker_transfer_reconstructor.pth"
SOURCE_DATA_PATH = "../VCTKProcessor/data/"
DEST_DATA_PATH = "data/"


with open(os.path.join(SOURCE_DATA_PATH, "conv_options.pkl"), 'rb') as f:
    conv_options = pickle.load(f)

example_tensor = torch.tensor(0.0)
if torch.cuda.is_available():
    example_tensor = example_tensor.cuda()


describer = util.load_model(DESCRIBER_NAME)
describer = Describer(describer, CATEG_SIZE)
describer.load_state_dict(torch.load(DESCRIBER_SNAPSHOT_PATH))
describer.eval()

reconstructor = util.load_model(RECONSTRUCTOR_NAME)
reconstructor = Reconstructor(reconstructor)
reconstructor.load_state_dict(torch.load(RECONSTRUCTOR_SNAPSHOT_PATH))
reconstructor.eval()

if example_tensor.is_cuda:
    describer = describer.cuda()
    reconstructor = reconstructor.cuda()


for speaker in range(SPEAKER_START_INDEX, NUM_SPEAKERS):
    path = os.path.join(
        SOURCE_DATA_PATH, "speech_" + str(speaker) + ".npy")
    speech = np.load(path)

    path = os.path.join(
        SOURCE_DATA_PATH, "sizes_" + str(speaker) + ".npy")
    sizes = np.load(path)

    num_utterances = sizes.shape[0]
    indices = np.concatenate([[0], np.cumsum(sizes)])

    speech_list = []
    for utterance in range(num_utterances):
        start_index = indices[utterance]
        end_index = indices[utterance + 1]

        orig = to_torch(conversions.to_log(
            speech[start_index:end_index], conv_options), example_tensor)

        reconst = from_torch(reconstructor.reconst(
            *describer.latent(orig)))

        speech_list.append(reconst)

    speech = np.concatenate(speech_list, axis=0)

    path = os.path.join(
        DEST_DATA_PATH, "speech_" + str(speaker) + ".npy")
    np.save(path, speech)

    path = os.path.join(
        DEST_DATA_PATH, "sizes_" + str(speaker) + ".npy")
    np.save(path, sizes)

with open(os.path.join(DEST_DATA_PATH, "conv_options.pkl"), 'wb') as f:
    pickle.dump(conv_options, f)

with open(os.path.join(SOURCE_DATA_PATH, "metadata.pkl"), 'rb') as f:
    metadata = pickle.load(f)

with open(os.path.join(DEST_DATA_PATH, "metadata.pkl"), 'wb') as f:
    pickle.dump(metadata, f)

band_mags = np.load(os.path.join(SOURCE_DATA_PATH, "band_mags.npy"))
np.save(os.path.join(DEST_DATA_PATH, "band_mags.npy"), band_mags)
