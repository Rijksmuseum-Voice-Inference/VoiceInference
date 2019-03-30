#!/usr/bin/env python3

import os
import pickle
import torch
import util
import conversions
from loader import to_torch

ENCODER_NAME = "speaker_features_encoder"
ENCODED_DIM = 128
ENCODER_SNAPSHOT_PATH = "snapshots/speaker_features_encoder.pth"
SAMPLE_RATE = 16000
DATA_PATH = "../VCTKProcessor/data"
INPUT_PATH = "../InferenceFiles/BBRecordings/trimmed"
OUTPUT_PATH = "../InferenceFiles/BBRecordings/bb_d_vector.pth"


with open(os.path.join(DATA_PATH, "conv_options.pkl"), 'rb') as f:
    conv_options = pickle.load(f)

example_tensor = torch.tensor(0.0)
if torch.cuda.is_available():
    example_tensor = example_tensor.cuda()


encoder = util.load_model(ENCODER_NAME)
encoder.load_state_dict(torch.load(ENCODER_SNAPSHOT_PATH))
encoder.eval()
if example_tensor.is_cuda:
    encoder = encoder.cuda()

result = torch.zeros(1, ENCODED_DIM)

files = [f for f in os.listdir(INPUT_PATH) if f.find('.wav') != -1]
encoded = example_tensor.new_zeros(len(files), ENCODED_DIM)

print("Found %d files" % len(files))

for (i, file) in enumerate(files):
    path = os.path.join(INPUT_PATH, file)
    speech, _ = conversions.encode(
        SAMPLE_RATE, conversions.load_wav(path, SAMPLE_RATE), conv_options)

    value = to_torch(conversions.to_log(speech, conv_options), example_tensor)
    encoded[i] = encoder(value).detach()

eligible_set = torch.arange(len(files))
while eligible_set.size()[0] > 1:
    num_eligible = eligible_set.size()[0]
    encoded_subset = encoded[eligible_set]

    encoded_mean = encoded_subset.mean(dim=0, keepdim=True)
    sq_distances = ((encoded_subset - encoded_mean) ** 2).sum(dim=1)
    _, best_indices = torch.topk(
        sq_distances, num_eligible // 2, largest=False)
    eligible_set = eligible_set[best_indices]

result = encoded[eligible_set[0]].unsqueeze(0).cpu()

torch.save(result, OUTPUT_PATH)
