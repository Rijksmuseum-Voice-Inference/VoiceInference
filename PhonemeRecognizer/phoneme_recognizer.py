import numpy as np
import torch
from torch import nn

fbank = np.loadtxt('mel-fbank.txt')
fbank = torch.from_numpy(fbank).float().cuda()
C = 4.41941738242e-05

def preprocess(S, ctx_len):    
    feats = torch.exp(S.squeeze(0)) - C    
    feats = torch.mm(feats, fbank.transpose(0,1))
    feats = torch.log(feats)
    feats -= torch.mean(feats, 0)    

    nframes = feats.size(0)
    nfreq = feats.size(1)

    nctx_frames = 2*ctx_len+1

    padded = torch.cat([
            torch.zeros(ctx_len, nfreq, device=feats.device),
            feats,
            torch.zeros(ctx_len, nfreq, device=feats.device)
        ], dim=0)

    new_feats = torch.zeros(nframes, nfreq*nctx_frames, device=feats.device, dtype=feats.dtype)    
    for i in range(nctx_frames):
        new_feats[:,i*nfreq:(i+1)*nfreq] = padded[i:i+nframes]

    return new_feats

class Net(nn.Module):
    """docstring for MLP"""
    def __init__(self, 
                input_dim, 
                output_dim, 
                n_hidden, 
                width,
                ctx_len,
                activation=nn.ReLU(),
                BN=False,
                downsample=-1):
        super(Net, self).__init__()

        modules = []
        for i in range(n_hidden):
            if i == 0:
                if downsample < 0:
                    modules.append(nn.Linear(input_dim, width))
                else:
                    modules.append(nn.Linear(input_dim, downsample))
                    modules.append(nn.Linear(downsample, width))
            else:
                modules.append(nn.Linear(width, width))
            if BN:
                modules.append(nn.BatchNorm1d(width))
            modules.append(activation)

        if n_hidden == 0:
            width = input_dim

        self.hidden = nn.Sequential(*modules)
        self.output = nn.Linear(width, output_dim)
        self.start = 0
        self.end = len(self.hidden)
        self.ctx_len = ctx_len

    def forward(self, x): 
        x = preprocess(x, self.ctx_len)       
        for i in range(self.start, self.end):
            x = self.hidden[i](x)
        x = self.output(x)
        return x

model = Net(440, 138, 3, 2048, 5, BN=True)
