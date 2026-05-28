import torch
import torch.nn as nn
import torch.fft
from models.TestModel.TimeBinPool import make_t_sec
from models.TestModel.FourierBasisPool import segment_fourier_pool
from models.modules.FourierPoolingAttentionHead import FourierPoolingAttentionHead

def chebyshev_transform(values, axis=-1):
    device = values.device
    
    if axis != -1 and axis != values.ndim - 1:
        perm = list(range(values.ndim))
        target_axis = perm.pop(axis)
        perm.append(target_axis)
        values = values.permute(*perm)
        
    N = values.shape[-1]
        
    if N > 1:
        x_part = torch.flip(values[..., 1:-1], dims=[-1])
        fft_input = torch.cat([values, x_part], dim=-1)
    else:
        fft_input = values
        
    coeffs = torch.fft.rfft(fft_input, dim=-1).real / (N - 1)
    
    coeffs[..., 0] /= 2
    if N > 1:
        coeffs[..., -1] /= 2
        
    k = torch.arange(N, device=device).float()
    signs = (-1.0) ** k
    coeffs = coeffs * signs
    
    if axis != -1 and axis != values.ndim - 1:
        inv_perm = [0] * values.ndim
        for cur_i, orig_i in enumerate(perm):
            inv_perm[orig_i] = cur_i
        coeffs = coeffs.permute(*inv_perm)

    return coeffs

def inverse_chebyshev_transform(coeffs, axis=-1, n_out=None):
    device = coeffs.device
    if axis != -1 and axis != coeffs.ndim - 1:
        perm = list(range(coeffs.ndim))
        target_axis = perm.pop(axis)
        perm.append(target_axis)
        coeffs = coeffs.permute(*perm)
        
    N_in = coeffs.shape[-1]
    N = n_out if n_out is not None else N_in
    
    temp_coeffs = coeffs
    if N > N_in:
        padding = torch.zeros(coeffs.shape[:-1] + (N - N_in,), device=device)
        temp_coeffs = torch.cat([coeffs, padding], dim=-1)
    elif N < N_in:
        temp_coeffs = coeffs[..., :N]
    
    k = torch.arange(N, device=device)
    signs = (-1.0) ** k
    scaled_coeffs = temp_coeffs * signs
    
    scaled_coeffs[..., 0] *= 2
    if N > 1:
        scaled_coeffs[..., -1] *= 2
        
    scaled_coeffs *= (N - 1)
    
    if N > 1:
        c_part = torch.flip(scaled_coeffs[..., 1:-1], dims=[-1])
        rfft_input = torch.cat([scaled_coeffs, c_part], dim=-1)
    else:
        rfft_input = scaled_coeffs
        
    values = torch.fft.rfft(rfft_input, dim=-1).real / (N - 1) / 2.0
    
    if axis != -1 and axis != coeffs.ndim - 1:
        inv_perm = [0] * coeffs.ndim
        for cur_i, orig_i in enumerate(perm):
            inv_perm[orig_i] = cur_i
        values = values.permute(*inv_perm)
        
    return values


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        
        self.mode_mixing = nn.Linear(n_modes, n_modes, bias=False)
        self.channel_mixing = nn.Linear(in_channels, out_channels, bias=False)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(1, out_channels, n_modes))
        else:
            self.register_parameter('bias', None)
            
        nn.init.xavier_uniform_(self.mode_mixing.weight)
        nn.init.xavier_uniform_(self.channel_mixing.weight)

    def forward(self, x_coeffs):
        x = self.mode_mixing(x_coeffs) 
        
        x = x.permute(0, 2, 1)
        x = self.channel_mixing(x)
        x = x.permute(0, 2, 1)
        
        if self.bias is not None:
            x = x + self.bias
            
        return x

class SNO1d(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 n_modes, 
                 hidden_channels=32, 
                 n_layers=4, 
                 **kwargs):
        super().__init__()
        self.n_modes = n_modes[0] if isinstance(n_modes, (list, tuple)) else n_modes
        self.n_layers = n_layers
        
        self.encoder = nn.Conv1d(in_channels, hidden_channels, 1)
        
        self.spectral_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.spectral_layers.append(
                SpectralConv1d(hidden_channels, hidden_channels, self.n_modes)
            )
            
        self.activation = nn.GELU()
        
        self.decoder = nn.Conv1d(hidden_channels, out_channels, 1)
        
    def forward(self, x, **kwargs):
        B, C, N = x.shape
        
        x_enc = self.encoder(x)
        
        coeffs = chebyshev_transform(x_enc, axis=-1)
        
        if N >= self.n_modes:
            coeffs_trunc = coeffs[..., :self.n_modes]
        else:
            padding = torch.zeros(B, x_enc.shape[1], self.n_modes - N, device=x.device)
            coeffs_trunc = torch.cat([coeffs, padding], dim=-1)
            
        for i, layer in enumerate(self.spectral_layers):
            coeffs_trunc = layer(coeffs_trunc)
            if i < self.n_layers - 1:
                coeffs_trunc = self.activation(coeffs_trunc)
                
        x_spec = inverse_chebyshev_transform(coeffs_trunc, axis=-1, n_out=N)
        
        x_res = x_spec + x_enc
        
        x_out = self.decoder(x_res)
        
        return x_out
SNO = SNO1d
        

class SNOModel(torch.nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels=128):
        super(SNOModel, self).__init__()
        
        self.sno = SNO(
            in_channels=input_channels,
            out_channels=32,
            n_modes=150,
            hidden_channels=hidden_channels,
            n_layers=4,
        )
        
        self.head = FourierPoolingAttentionHead(
            input_channels=32,
            output_dim=output_channels,
            num_heads=4,
        )

    def forward(self, x, Fs, **kwargs):
        x = self.sno(x)

        x = x.permute(0, 2, 1)
        
        pred_y = self.head(x, Fs)
        
        return pred_y