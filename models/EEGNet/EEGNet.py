import torch
from torch import nn
from braindecode.models import EEGNet

class EEGNetModel(nn.Module):
    def __init__(self, input_channels, num_classes, epoch_length, sampling_rate):
        super().__init__()
        self.eegnet = EEGNet(
            n_chans=input_channels,
            n_outputs=num_classes,
            n_times=epoch_length*sampling_rate
        )
    
    def forward(self, x, **kwargs):
        return self.eegnet(x)
