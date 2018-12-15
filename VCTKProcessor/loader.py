import os
import pickle
import threading
import numpy as np
import conversions


def from_torch(frames):
    return frames.detach().cpu().numpy()[0].T


def to_torch(frames, example_tensor):
    return example_tensor.new_tensor(frames.T[np.newaxis, :, :])


class VCTKLoader:
    def __init__(self, data_path, example_tensor, features,
                 num_speakers=100, speaker_start_index=0,
                 speaker_take_count=40, utterance_take_count=40):
        self.data_path = data_path
        self.example_tensor = example_tensor

        with open(os.path.join(data_path, 'conv_options.pkl'), 'rb') as f:
            self.conv_options = pickle.load(f)

        if features == 'direct':
            self.feature_extractor = lambda mag_frames: \
                to_torch(mag_frames, example_tensor)
        elif features == 'log':
            self.feature_extractor = lambda mag_frames: \
                to_torch(conversions.to_log(
                    mag_frames, self.conv_options), example_tensor)
        elif features == 'mag_norm':
            band_mags = np.load(os.path.join(data_path, 'band_mags.npy'))
            self.feature_extractor = lambda mag_frames: \
                to_torch(conversions.to_mag_norm(
                    mag_frames, band_mags, self.conv_options), example_tensor)
        elif features == 'two':
            band_mags = np.load(os.path.join(data_path, 'band_mags.npy'))
            self.feature_extractor = lambda mag_frames: \
                to_torch(conversions.to_two(
                    mag_frames, band_mags, self.conv_options), example_tensor)
        else:
            raise RuntimeError("Invalid feature type: " + features)

        self.num_speakers = num_speakers
        self.speaker_start_index = speaker_start_index
        self.speaker_take_count = speaker_take_count
        self.utterance_take_count = utterance_take_count

    def __iter__(self):
        nonlocal_vars = {}
        nonlocal_vars['speech'] = np.zeros((0, 0))
        nonlocal_vars['indices'] = np.zeros(0)
        nonlocal_vars['speakers'] = np.zeros(0)
        nonlocal_vars['order'] = np.zeros(0)

        nonlocal_vars['speech_next'] = None
        nonlocal_vars['indices_next'] = None
        nonlocal_vars['speakers_next'] = None
        nonlocal_vars['order_next'] = None

        def iterate():
            for utterance in nonlocal_vars['order']:
                start_index = nonlocal_vars['indices'][utterance]
                end_index = nonlocal_vars['indices'][utterance + 1]

                result = self.feature_extractor(
                    nonlocal_vars['speech'][start_index:end_index])

                yield (result, nonlocal_vars['speakers'][utterance])

        def load_data():
            speech_list = []
            sizes_list = [np.zeros(1, dtype=np.int64)]
            speaker_list = []
            speaker_sample = np.random.choice(
                self.num_speakers,
                size=self.speaker_take_count,
                replace=False) + self.speaker_start_index

            for speaker in speaker_sample:
                speaker = speaker.item()

                path = os.path.join(
                    self.data_path, "speech_" + str(speaker) + ".npy")
                speech = np.load(path, mmap_mode='r')
                path = os.path.join(
                    self.data_path, "sizes_" + str(speaker) + ".npy")
                sizes = np.load(path)

                if self.utterance_take_count < sizes.shape[0]:
                    start = np.random.choice(
                        sizes.shape[0] - self.utterance_take_count + 1)
                    end = start + self.utterance_take_count
                    start_frames = np.sum(sizes[:start])
                    end_frames = start_frames + np.sum(sizes[start:end])
                    speech = speech[start_frames:end_frames]
                    sizes = sizes[start:end]

                speech_list.append(speech)
                sizes_list.append(sizes.astype(np.int64))
                speaker_list.append(np.zeros(
                    sizes.shape[0], dtype=np.int64) + speaker)

            nonlocal_vars['speech_next'] = \
                np.concatenate(speech_list, axis=0)
            nonlocal_vars['indices_next'] = \
                np.cumsum(np.concatenate(sizes_list), axis=0)
            nonlocal_vars['speakers_next'] = \
                np.concatenate(speaker_list)
            del(speech_list)

            num_utterances = nonlocal_vars['indices_next'].shape[0] - 1
            nonlocal_vars['order_next'] = np.random.permutation(num_utterances)

        while True:
            thread = threading.Thread(target=load_data)
            thread.start()

            for data in iterate():
                yield data

            nonlocal_vars['speech'] = None
            nonlocal_vars['indices'] = None
            nonlocal_vars['speakers'] = None
            nonlocal_vars['order'] = None

            thread.join()

            nonlocal_vars['speech'] = nonlocal_vars['speech_next']
            nonlocal_vars['indices'] = nonlocal_vars['indices_next']
            nonlocal_vars['speakers'] = nonlocal_vars['speakers_next']
            nonlocal_vars['order'] = nonlocal_vars['order_next']
