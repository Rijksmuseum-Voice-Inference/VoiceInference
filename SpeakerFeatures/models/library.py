import torch
from torch.nn import *
import torch.nn.functional as F


class Reshape(Module):
    def __init__(self, *target_size):
        super().__init__()
        self.target_size = target_size

    def forward(self, features):
        batch_size = features.size()[0]
        return features.reshape([batch_size, *self.target_size])


class Resize(Module):
    def __init__(self, *target_size):
        super().__init__()
        self.target_size = target_size

    def forward(self, features):
        return F.interpolate(features, self.target_size)


class RevertSize(Module):
    def __init__(self, sizes_list, value=0, transform={}):
        super().__init__()
        self.sizes_list = sizes_list
        self.value = value
        self.transform = transform

    def forward(self, features):
        sizes = self.sizes_list.pop()
        orig_size = features.size()
        result = features

        for dim, new_size in enumerate(sizes):
            if dim in self.transform:
                new_size = self.transform[dim]
                if new_size == -1:
                    continue
            if orig_size[dim] < new_size:
                target_size = list(result.size())
                target_size[dim] = new_size - orig_size[dim]
                result = torch.cat([
                    result, result.new_zeros(*target_size) + self.value],
                    dim=dim)
            elif orig_size[dim] > new_size:
                result = result.narrow(dim, 0, new_size)

        return result


class PadToMinimum(Module):
    def __init__(self, minimum_size, dim, value=0):
        super().__init__()
        self.minimum_size = minimum_size
        self.dim = dim
        self.value = value

    def forward(self, features):
        orig_size = features.size()[self.dim]
        if orig_size < self.minimum_size:
            target_size = list(features.size())
            target_size[self.dim] = self.minimum_size - orig_size
            features = torch.cat([
                features, features.new_zeros(*target_size) + self.value],
                dim=self.dim)
        return features


class GlobalAvgPool(Module):
    def forward(self, features):
        batch_size = features.size()[0]
        channels = features.size()[1]
        return features.reshape(batch_size, channels, -1).mean(dim=2)


class UndoGlobalAvgPool(Module):
    def __init__(self, sizes_list):
        super().__init__()
        self.sizes_list = sizes_list

    def forward(self, features):
        sizes = self.sizes_list.pop()
        batch_size = features.size()[0]
        channels = features.size()[1]
        return features.reshape(
            batch_size, channels, *([1] * len(sizes[2:]))).repeat(
            1, 1, *sizes[2:])


class PrintSize(Module):
    def forward(self, features):
        print(features.size())
        return features


class AppendSize(Module):
    def __init__(self, output_list):
        super().__init__()
        self.output_list = output_list

    def forward(self, features):
        self.output_list.append(features.size())
        return features
