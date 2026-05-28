import torch
import torch.nn as nn
from models.modules.FourierBasisPool import segment_fourier_pool


def make_t_sec(B: int, N: int, fs: float, device, dtype=torch.float32):
    t = torch.arange(N, device=device, dtype=dtype) / float(fs)
    return t.unsqueeze(0).expand(B, -1)  

class AttnPool(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        B = x.size(0)
        q = self.q.expand(B, -1, -1)
        y, _ = self.attn(q, x, x)
        return self.ln(y.squeeze(1))

class TokenHead(nn.Module):
    def __init__(self, d_model, out_dim, n_layers=2, num_heads=4, dropout=0.2):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pool = AttnPool(d_model, num_heads=num_heads)
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
    

class FourierPoolingAttentionHead(nn.Module):
    def __init__(self, input_channels, output_dim, num_heads=4, M=50, T_total=30, T_seg=5, batch_first=True):
        super(FourierPoolingAttentionHead, self).__init__()
        
        self.M = M
        self.T_total = T_total
        self.T_seg = T_seg
        
        self.attention = nn.MultiheadAttention(embed_dim=input_channels, num_heads=num_heads, batch_first=batch_first)
        self.head = TokenHead(input_channels, output_dim, num_heads=num_heads)

    def forward(self, x, fs):
        t_out_sec = make_t_sec(B=x.shape[0], N=x.shape[1], fs=fs, device=x.device)
        pooled = segment_fourier_pool(x, t_out_sec, M=self.M, T_total=self.T_total, T_seg=self.T_seg)
        pooled, _ = self.attention(pooled, pooled, pooled)
        pred_y = self.head(pooled)
        return pred_y