import torch
import torch.nn as nn
from speaker_id import ResNet

MODEL_FILE = 'model.pt'

def get_model():
    net = ResNet(1, 4, 16, 64, 256, 1211)
    model = nn.DataParallel(net, device_ids=[0]).cuda()
    model.load_state_dict(torch.load( MODEL_FILE ))
    model.eval()
    return model
get_model()
    
