import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys, math
import numpy as np
from torch.utils.data import Dataset, DataLoader
from numpy.matlib import repmat

VAD_THRESHOLD = -9.

class MelBankDataset(Dataset):

    def __init__(self, file_list, target_list, st_index, end_index):
        self.file_list = file_list
        self.target_list = target_list
        self.n_class = len(list(set(target_list)))
        self.st_index = st_index
        self.end_index = end_index

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        mbk = np.load(self.file_list[index], mmap_mode='r')[self.st_index[index]:self.end_index[index]]
        B = repmat(mbk, int(math.ceil(16383.0/(mbk.shape[0]))), 1)
        D = B[:16383,:][np.newaxis, ...]
        label = self.target_list[index]
        D = D.astype('float32')
        return D, label

def parse_data():
    mbkfile_list = []
    ID_list = []
    st_index = []
    end_index = []
    all_data = open('voxceleb_train.csv', 'r').readlines()
    mbkfile_list = map(lambda x: x.split()[0], all_data)
    st_index = map(lambda x: int(x.split()[1]), all_data)
    end_index = map(lambda x: int(x.split()[2]), all_data)
    ID_list = map(lambda x: x.split()[3], all_data)
            
    # construct a dictionary, where key and value are correspond to ID and target
    uniqueID_list = list(set(ID_list))
    class_n = len(uniqueID_list)
    target_dict = dict(zip(uniqueID_list, range(class_n)))
    label_list = [target_dict[ID_key] for ID_key in ID_list]

    print '#mbkfile  #label  #frame(min) #frame(max)'
    print len(mbkfile_list), '   ', len(list(set(label_list))) 
    return mbkfile_list, label_list, st_index, end_index, class_n


class BasicBlock(nn.Module):
    def __init__(self, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

## construct network
class ResNet(nn.Module):
    def __init__(self, channel_0, channel_1, channel_2, channel_3, channel_4, class_n):
        super(ResNet, self).__init__()
        self.featureD = 128
        self.convlayers = nn.Sequential(
            nn.Conv2d(channel_0, channel_1, 3, 2, bias=False),
            nn.BatchNorm2d(channel_1),
            nn.ReLU(inplace=True),
            BasicBlock(channel_1),
            BasicBlock(channel_1),
            nn.Conv2d(channel_1, channel_2, 3, 2, bias=False),
            nn.BatchNorm2d(channel_2),
            nn.ReLU(inplace=True),
            BasicBlock(channel_2),
            BasicBlock(channel_2),
            nn.Conv2d(channel_2, channel_3, 3, 2, bias=False),
            nn.BatchNorm2d(channel_3),
            nn.ReLU(inplace=True),
            BasicBlock(channel_3),
            BasicBlock(channel_3),
            nn.Conv2d(channel_3, channel_4, 3, 2, bias=False),
            nn.BatchNorm2d(channel_4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_4, self.featureD, 3, 2, bias=False),
        )
        self.bn = nn.BatchNorm1d(self.featureD, affine=False)
        self.fc1 = nn.utils.weight_norm(nn.Linear(self.featureD, class_n, bias=False), name='weight')
        
    def apply_vad(self,x):
      
        # x = x_inp
        batch_size = x.shape[0]
        time_dim = x.shape[2]
        freq_dim = x.shape[3]
        out_time_dim = time_dim
        # print x_inp.shape #(B, 1, T, F)  (5L, 1L, 16383L, 258L)
        means = torch.mean(x, dim=3)
        xMap = means > VAD_THRESHOLD # (B, T) (5L, 16383L)
        M = Variable(torch.zeros((batch_size, time_dim, time_dim))).cuda()
             
        for i in range(batch_size):
            xMap_ins = xMap[i].view(-1,)
            x_axis = xMap_ins.nonzero().view(-1,)
            curr_shape = x_axis.shape[0]
            y_axis = torch.Tensor(range(curr_shape)).cuda().view(-1,)
            M_ins = Variable(torch.zeros(time_dim, curr_shape)).cuda()
            M_ins[x_axis.data.long(), y_axis.long()] = 1
            M_ins = M_ins[:,:curr_shape]
            M_ins = M_ins.repeat(1,int(math.ceil(time_dim/float(curr_shape))))[:,:time_dim]
            M[i] = M_ins

        x = torch.transpose(torch.bmm(torch.transpose(x.squeeze(),1,2), M), 1,2).unsqueeze(1)
        return x

    def forward(self, x):
        x = self.apply_vad(x)
        conv_o = self.convlayers(x)
        x = F.avg_pool2d(conv_o, [conv_o.size()[2], conv_o.size()[3]], stride=1)
        x = x.view(-1, self.featureD)
        x = self.bn(x)
        return x, F.log_softmax(self.fc1(x))

