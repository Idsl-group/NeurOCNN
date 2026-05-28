import math
import torch


def dt_from_tsec(t_sec: torch.Tensor) -> torch.Tensor:
    dt = t_sec[:, 1:] - t_sec[:, :-1]
    dt_last = dt[:, -1:].clamp_min(1e-6)
    return torch.cat([dt, dt_last], dim=1)

def fourier_basis(t: torch.Tensor, M: int, T: float):
    B, N = t.shape
    device, dtype = t.device, t.dtype
    phi = torch.zeros(B, N, M, device=device, dtype=dtype)

    phi[..., 0] = 1.0
    m = 1
    k = 1
    while m < M:
        w = 2.0 * math.pi * k / float(T)
        if m < M:
            phi[..., m] = torch.sin(w * t); m += 1
        if m < M:
            phi[..., m] = torch.cos(w * t); m += 1
        k += 1
    return phi

def basis_project_pool(x_tok: torch.Tensor, t_sec: torch.Tensor, M: int, T: float):
    B, N, d = x_tok.shape
    t0 = t_sec[:, :1]
    t = (t_sec - t0).clamp(0.0, float(T))
    dt = dt_from_tsec(t_sec).to(x_tok.dtype).clamp_min(1e-8)

    phi = fourier_basis(t, M=M, T=T)
    phi_w = phi * dt.unsqueeze(-1)

    c = torch.einsum("bnm,bnd->bmd", phi_w, x_tok)
    return c

def segment_fourier_pool(x_tok, t_sec, M, T_total, T_seg):
    B, N, d = x_tok.shape
    n_seg = int(round(T_total / T_seg))
    fs = 1.0 / (t_sec[:, 1] - t_sec[:, 0]).mean().item()
    seg_len = int(round(T_seg * fs))
    seg_len = max(seg_len, 2)

    coeffs = []
    for i in range(n_seg):
        s = i * seg_len
        e = min((i + 1) * seg_len, N)
        x_i = x_tok[:, s:e, :]
        t_i = t_sec[:, s:e]
        coeffs.append(basis_project_pool(x_i, t_i, M=M, T=T_seg))

    return torch.cat(coeffs, dim=1)