import torch
from torch.nn import *
import torch.nn.functional as F


class MultConst(Module):
    def __init__(self, multiplier):
        super().__init__()
        self.multiplier = multiplier

    def forward(self, features):
        return features * self.multiplier


class AddConst(Module):
    def __init__(self, addend):
        super().__init__()
        self.addend = addend

    def forward(self, features):
        return features + self.addend


class LearnableBias(Module):
    def __init__(self):
        super().__init__()
        self.bias = Parameter(torch.zeros(1))

    def forward(self, features):
        return features + self.bias


class Dropout1d(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dropout2d = Dropout2d(*args, **kwargs)

    def forward(self, features):
        return self.dropout2d(
            features.unsqueeze(dim=3)).reshape(*features.size())


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


class Slice(Module):
    def __init__(self, size, dim):
        super().__init__()
        self.size = size
        self.dim = dim

    def forward(self, features):
        return features.narrow(self.dim, 0, self.size)


class Transpose(Module):
    def __init__(self, dim_0, dim_1):
        super().__init__()
        self.dim_0 = dim_0
        self.dim_1 = dim_1

    def forward(self, features):
        return torch.transpose(features, self.dim_0, self.dim_1)


class TupleSelector(Module):
    def __init__(self, module, index):
        super().__init__()
        self.module = module
        self.index = index

    def forward(self, features):
        return self.module(features)[self.index]


class PadToMinimum(Module):
    def __init__(self, minimum_size, dim):
        super().__init__()
        self.minimum_size = minimum_size
        self.dim = dim

    def forward(self, features):
        orig_size = features.size()[self.dim]
        if orig_size < self.minimum_size:
            padding_size = [1] * features.dim()
            padding_size[self.dim] = self.minimum_size - orig_size
            last_index = features.new_tensor(orig_size - 1, dtype=torch.long)
            padding_value = features.index_select(self.dim, last_index)
            padding = padding_value.repeat(*padding_size)
            features = torch.cat([features, padding], dim=self.dim)
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


class PartialAvgPool(Module):
    def __init__(self, *num_average):
        super().__init__()
        self.num_average = num_average
        self.total_count = sum(self.num_average)
        self.avg_pool = GlobalAvgPool()

    def forward(self, features):
        batch_size, channel_size = features.size()[0:2]
        results = []
        pos = channel_size - self.total_count

        for count in self.num_average:
            results.append(self.avg_pool(features[:, pos:(pos + count)]))
            pos += count

        return (features[:, :-self.total_count], *results)


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
