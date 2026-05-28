import torch
from neuralop.layers.local_no_block import LocalNOBlocks
from models.modules.FourierPoolingAttentionHead import FourierPoolingAttentionHead
        

class LNO(torch.nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels=128, N=3000):
        super(LNO, self).__init__()

        self.lno = LocalNOBlocks(
            in_channels=input_channels,
            out_channels=32,
            n_modes=(150,),
            default_in_shape=(N,),
            disco_layers=False,
            n_layers=4,
        )
        
        self.head = FourierPoolingAttentionHead(
            input_channels=32,
            output_dim=output_channels,
            num_heads=2,
        )

    def forward(self, x, Fs, **kwargs):
        x = self.lno(x)
        fs = Fs

        x = x.permute(0, 2, 1)
        
        pred_y = self.head(x, fs)
        
        return pred_y