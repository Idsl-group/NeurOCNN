import torch
from neuralop.models import UNO
from models.modules.FourierPoolingAttentionHead import FourierPoolingAttentionHead
        

class UNOModel(torch.nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels=32):
        super(UNOModel, self).__init__()
        
        self.uno = UNO(
            in_channels=input_channels,
            out_channels=32,
            hidden_channels=hidden_channels,
            n_layers=4,
            positional_embedding="grid",
            uno_out_channels=[32, 64, 64, 32],
            uno_n_modes=[[150], [150], [150], [150]],
            uno_scalings=[[1.0], [1.0], [1.0], [1.0]],
            channel_mlp_skip=None
        )
        
        self.head = FourierPoolingAttentionHead(
            input_channels=32,
            output_dim=output_channels,
            num_heads=4,
        )

    def forward(self, x, Fs, **kwargs):
        x = self.uno(x)
        fs = Fs

        x = x.permute(0, 2, 1)
        
        pred_y = self.head(x, fs)
        return pred_y