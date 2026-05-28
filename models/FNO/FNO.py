import torch
import torch.nn as nn
from neuralop.models import FNO
from models.modules.FourierPoolingAttentionHead import FourierPoolingAttentionHead
        

class FNOModel(torch.nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels=128):
        super(FNOModel, self).__init__()
        
        self.fno = FNO(
            n_modes=(150,),
            in_channels=input_channels,
            out_channels=32,
            hidden_channels=hidden_channels,
            n_layers=4,
            positional_embedding="grid",
        )
        
        self.head = FourierPoolingAttentionHead(
            input_channels=32,
            output_dim=output_channels,
            num_heads=4,
        )

    def forward(self, x, Fs, **kwargs):
        x = self.fno(x)
        fs = Fs

        x = x.permute(0, 2, 1)
        
        pred_y = self.head(x, fs)
        
        return pred_y