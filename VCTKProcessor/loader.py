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
        speech = np.zeros((0, 0))
        indices = np.zeros(0)
        speakers = np.zeros(0)
        order = np.zeros(0)

        speech_next = None
        indices_next = None
        speakers_next = None
        order_next = None

        def iterate():
            nonlocal speech
            nonlocal indices
            nonlocal speakers
            nonlocal order

            for utterance in order:
                start_index = indices[utterance]
                end_index = indices[utterance + 1]

                result = self.feature_extractor(speech[start_index:end_index])

                yield (result, speakers[utterance])

        def load_data():
            nonlocal speech_next
            nonlocal indices_next
            nonlocal speakers_next
            nonlocal order_next

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

            speech_next = np.concatenate(speech_list, axis=0)
            indices_next = np.cumsum(np.concatenate(sizes_list), axis=0)
            speakers_next = np.concatenate(speaker_list)
            del(speech_list)

            num_utterances = indices_next.shape[0] - 1
            order_next = np.random.permutation(num_utterances)

        while True:
            thread = threading.Thread(target=load_data)
            thread.start()

            yield from iterate()

            speech = None
            indices = None
            speakers = None
            order = None

            thread.join()

            speech = speech_next
            indices = indices_next
            speakers = speakers_next
            order = order_next
