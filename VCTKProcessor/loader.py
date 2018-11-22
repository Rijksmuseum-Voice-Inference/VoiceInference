import os
import threading
import numpy as np
import torch
import torch.utils.data


class VCTKLoader:
    def __init__(self, data_path, example_tensor,
                 num_speakers=100, speaker_start_index=0,
                 speaker_take_count=40, utterance_take_count=40):
        self.data_path = data_path
        self.example_tensor = example_tensor

        self.num_speakers = num_speakers
        self.speaker_start_index = speaker_start_index
        self.speaker_take_count = speaker_take_count
        self.utterance_take_count = utterance_take_count

    def __iter__(self):
        speech = torch.zeros(0, 0)
        indices = torch.zeros(0)
        speakers = torch.zeros(0)
        order = torch.zeros(0)

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
                utterance = utterance.item()
                start_index = indices[utterance].item()
                end_index = indices[utterance + 1].item()

                result = torch.t(speech[start_index:end_index])
                result = result.reshape([1, *result.size()])
                result = self.example_tensor.new_tensor(result)

                yield (result, speakers[utterance].item())

        def load_data():
            nonlocal speech_next
            nonlocal indices_next
            nonlocal speakers_next
            nonlocal order_next

            speech_list = []
            sizes_list = [torch.zeros(1, dtype=torch.int64)]
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

                speech = torch.tensor(speech)
                sizes = torch.tensor(sizes, dtype=torch.int64)

                speech_list.append(speech)
                sizes_list.append(sizes)
                speaker_list.append(torch.zeros(
                    sizes.size()[0], dtype=torch.int64) + speaker)

            speech_next = torch.cat(speech_list, dim=0)
            indices_next = torch.cumsum(torch.cat(sizes_list), dim=0)
            speakers_next = torch.cat(speaker_list)
            del(speech_list)

            num_utterances = indices_next.size()[0] - 1
            order_next = torch.arange(num_utterances)[
                torch.randperm(num_utterances)]

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
