import torch
import torch.nn as nn
from models.modules.SplineConv1d import SplineConv1d
from models.modules.FourierPoolingAttentionHead import FourierPoolingAttentionHead
from models.modules.CKConv import CKConv


class AttnPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        B = x.size(0)
        q = self.q.expand(B, -1, -1)
        y, _ = self.attn(q, x, x)
        return self.ln(y.squeeze(1))

class TokenHead(nn.Module):
    def __init__(self, d_model, out_dim, n_layers=2, dropout=0.2):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pool = AttnPool(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2*d_model, out_dim),
        )

    def forward(self, tok):
        tok = self.enc(tok)
        z = self.pool(tok)
        return self.mlp(z)
    
        
class NeurOCNN(torch.nn.Module):
    def __init__(
        self, 
        input_channels, 
        hidden_channels, 
        output_channels, 
        kernel_duration=0.5, 
        num_ctrl_points=15, 
        convolution_type="spline",
        M=50,
        T_total=30,
        T_seg=5.0,
        ):
        super(NeurOCNN, self).__init__()
        
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        
        if convolution_type == "spline":
            self.conv_1 = SplineConv1d(
                in_channels=input_channels,
                out_channels=64,
                kernel_duration=kernel_duration,
                num_ctrl_points=num_ctrl_points,
                padding="same",
            )
            self.conv_2 = SplineConv1d(
                in_channels=64,
                out_channels=128,
                kernel_duration=kernel_duration,
                num_ctrl_points=num_ctrl_points,
                padding="same",
            )
            self.conv_3 = SplineConv1d(
                in_channels=128,
                out_channels=64,
                kernel_duration=kernel_duration,
                num_ctrl_points=num_ctrl_points,
                padding="same",
            )
            self.conv_4 = SplineConv1d(
                in_channels=64,
                out_channels=32,
                kernel_duration=kernel_duration,
                num_ctrl_points=num_ctrl_points,
                padding="same",
            )
        else:
            self.conv_1 = CKConv(
                in_channels=input_channels,
                out_channels=64,
                kernel_duration=kernel_duration,
                padding="same",
                stride=None, # stride = 1 sample
                hidden_channels=32,
                activation_function="Sine",
                norm_type="",
                omega_0=30.0,
                weight_dropout=0.1,
                bias=True,
            )
            self.conv_2 = CKConv(
                in_channels=64,
                out_channels=128,
                kernel_duration=kernel_duration,
                padding="same",
                stride=None, # stride = 1 sample
                hidden_channels=32,
                activation_function="Sine",
                norm_type="",
                omega_0=30.0,
                weight_dropout=0.1,
                bias=True,
            )
            self.conv_3 = CKConv(
                in_channels=128,
                out_channels=64,
                kernel_duration=kernel_duration,
                padding="same",
                stride=None, # stride = 1 sample
                hidden_channels=32,
                activation_function="Sine",
                norm_type="",
                omega_0=30.0,
                weight_dropout=0.1,
                bias=True,
            )
            self.conv_4 = CKConv(
                in_channels=64,
                out_channels=32,
                kernel_duration=kernel_duration,
                padding="same",
                stride=None, # stride = 1 sample
                hidden_channels=32,
                activation_function="Sine",
                norm_type="",
                omega_0=30.0,
                weight_dropout=0.1,
                bias=True,
            )
            
            
        self.layer_norm = nn.GroupNorm(num_groups=1, num_channels=64)
        self.gelu = nn.GELU()
        
        self.layer_norm_2 = nn.GroupNorm(num_groups=1, num_channels=128)
        self.gelu_2 = nn.GELU()
        
        self.layer_norm_3 = nn.GroupNorm(num_groups=1, num_channels=64)
        self.gelu_3 = nn.GELU()
        
        self.layer_norm_4 = nn.GroupNorm(num_groups=1, num_channels=32)
        
        self.head = FourierPoolingAttentionHead(
            input_channels=32,
            output_dim=output_channels,
            num_heads=4,
            M=M,
            T_total=T_total,
            T_seg=T_seg,
        )

    def forward(self, x, Fs, **kwargs):
        x, fs = self.conv_1(x, Fs)
        x = self.gelu(x)
        x = self.layer_norm(x)
        x, fs = self.conv_2(x, fs)
        x = self.gelu_2(x)
        x = self.layer_norm_2(x)
        x, fs = self.conv_3(x, fs)
        x = self.gelu_3(x)
        x = self.layer_norm_3(x)
        x, fs = self.conv_4(x, fs)        
        x = self.layer_norm_4(x)

        x = x.permute(0, 2, 1)
        pred_y = self.head(x, fs)
        return pred_y


