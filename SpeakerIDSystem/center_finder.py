import os
import pickle
import numpy as np
import torch
import util
import conversions
from loader import to_torch

NUM_SPEAKERS = 100
SPEAKER_START_INDEX = 0
ENCODER_NAME = "speaker_id_model"
ENCODED_DIM = 1211
ENCODER_SNAPSHOT_PATH = "model.pt"
DATA_PATH = "../VCTKProcessor/data/"
CENTERS_PATH = "data/centers.pth"


with open(os.path.join(DATA_PATH, "conv_options.pkl"), 'rb') as f:
    conv_options = pickle.load(f)

example_tensor = torch.tensor(0.0)
if torch.cuda.is_available():
    example_tensor = example_tensor.cuda()


encoder = util.load_model(ENCODER_NAME, "", True)
encoder.load_state_dict(torch.load(ENCODER_SNAPSHOT_PATH))
encoder.eval()
if example_tensor.is_cuda:
    encoder = encoder.cuda()

result = torch.zeros(NUM_SPEAKERS, ENCODED_DIM)
total_cluster_score = 0.0
total_discrim_score = 0.0

for speaker in range(SPEAKER_START_INDEX, NUM_SPEAKERS):
    path = os.path.join(
        DATA_PATH, "speech_" + str(speaker) + ".npy")
    speech = np.load(path)

    path = os.path.join(
        DATA_PATH, "sizes_" + str(speaker) + ".npy")
    sizes = np.load(path)

    num_utterances = sizes.shape[0]
    indices = np.concatenate([[0], np.cumsum(sizes)])

    encoded = example_tensor.new_zeros(num_utterances, ENCODED_DIM)
    for utterance in range(num_utterances):
        start_index = indices[utterance]
        end_index = indices[utterance + 1]

        value = to_torch(conversions.to_log(
            speech[start_index:end_index], conv_options), example_tensor)

        _, output = encoder(value.unsqueeze(0))
        encoded[utterance] = output[0].detach()

    eligible_set = torch.arange(num_utterances)
    while eligible_set.size()[0] > 1:
        num_eligible = eligible_set.size()[0]
        encoded_subset = encoded[eligible_set]

        encoded_mean = encoded_subset.mean(dim=0, keepdim=True)
        sq_distances = ((encoded_subset - encoded_mean) ** 2).sum(dim=1)
        _, best_indices = torch.topk(
            sq_distances, num_eligible // 2, largest=False)
        eligible_set = eligible_set[best_indices]

    center = encoded[eligible_set[0]].unsqueeze(0)
    result[speaker] = center[0]

    cluster_score = ((encoded - center) ** 2).mean().item()
    print("(%0.3f)" % cluster_score,
          end=' ', flush=True)
    total_cluster_score += cluster_score

torch.save(result, CENTERS_PATH)
print('')
print("Cluster score: %0.3f" % (total_cluster_score / NUM_SPEAKERS))