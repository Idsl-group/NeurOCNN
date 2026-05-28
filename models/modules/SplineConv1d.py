import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline


class SplineConv1d(nn.Module):
    """
    Continuous-time convolution using a spline-based kernel.
    Kernel is sampled based on input sampling rate fs.

    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        kernel_duration: duration of the kernel in seconds
        num_ctrl_points: number of control points for the spline
        padding: padding to be applied to the input (same, valid or int)
        stride: stride to be applied to the input (in seconds. None => 1 sample)

    Inputs:
        x: input tensor (B, C_in, N_in)
        fs: sampling rate of the input (scalar or tensor)

    Returns:
        out: output of the convolution (B, C_out, N_out)
        fs_out: output sampling rate (scalar or tensor)
    """

    def __init__(self, in_channels, out_channels,
                 kernel_duration=0.15,   # in seconds
                 num_ctrl_points=16,
                 padding="same",         # "same", "valid" or int
                 stride=None):           # in seconds; None => 1 sample
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_duration = float(kernel_duration)
        self.num_ctrl_points = int(num_ctrl_points)
        self.padding = padding

        self.stride = None if stride is None else float(stride)
        if self.stride is not None and self.stride < 0:
            raise ValueError("stride must be >= 0 seconds or None.")

        # learnable spline control points: (out, in, num_ctrl_points)
        self.ctrl = nn.Parameter(
            torch.randn(out_channels, in_channels, num_ctrl_points)
        )

        # fixed knot locations (normalized time)
        ctrl_t = torch.linspace(-0.5, 0.5, num_ctrl_points)
        self.register_buffer("ctrl_t", ctrl_t)

        # cache interpolation matrices
        self._interp_cache = {}

    # get interpolation matrix
    def _get_interp_matrix(self, kernel_len: int, device, dtype):
        key = (kernel_len, dtype)
        if key in self._interp_cache:
            A = self._interp_cache[key]
            return A.to(device) if A.device != device else A

        with torch.no_grad():
            t_real = torch.linspace(-0.5, 0.5, kernel_len, device=device, dtype=dtype)
            y_basis = torch.eye(self.num_ctrl_points, device=device, dtype=dtype)

            ctrl_t = self.ctrl_t.to(device=device, dtype=dtype)
            coeffs = natural_cubic_spline_coeffs(ctrl_t, y_basis)
            spline = NaturalCubicSpline(coeffs)

            A = spline.evaluate(t_real)

        A = A.detach()
        self._interp_cache[key] = A
        return A

    # kernel sampling
    def sample_kernel(self, fs: float):
        if isinstance(fs, torch.Tensor):
            fs = fs.item()
        fs = float(fs)

        kernel_len = int(round(self.kernel_duration * fs))
        kernel_len = max(kernel_len, 1)
        if kernel_len % 2 == 0:
            kernel_len += 1

        device = self.ctrl.device
        dtype = self.ctrl.dtype

        A = self._get_interp_matrix(kernel_len, device, dtype)

        y = self.ctrl.view(self.out_channels * self.in_channels, self.num_ctrl_points)
        k_flat = torch.matmul(y, A.T)
        kernel = k_flat.view(self.out_channels, self.in_channels, kernel_len)
        return kernel

    # stride conversion (seconds to samples)
    def _stride_samples(self, fs_val: float) -> int:
        if self.stride is None:
            return 1
        s = int(round(self.stride * fs_val))
        return max(s, 1)

    # "same" padding for arbitrary stride
    @staticmethod
    def _pad_same_1d(x: torch.Tensor, kernel_len: int, stride: int, dilation: int = 1):
        L = x.size(-1)
        out_len = (L + stride - 1) // stride  # ceil(L/stride)
        effective_k = dilation * (kernel_len - 1) + 1
        pad_total = max((out_len - 1) * stride + effective_k - L, 0)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        if pad_total > 0:
            x = F.pad(x, (pad_left, pad_right))
        return x

    def forward(self, x, fs):
        if isinstance(fs, torch.Tensor):
            fs_flat = fs.view(-1)
            if fs_flat.numel() > 1:
                if not torch.allclose(fs_flat, fs_flat[0].expand_as(fs_flat)):
                    raise ValueError(
                        "SplineKernelConv1d expects one sampling rate per batch, "
                        f"but got multiple values: {fs_flat}"
                    )
            fs_val = float(fs_flat[0].item())
        else:
            fs_val = float(fs)

        kernel = self.sample_kernel(fs_val).to(x.device, x.dtype)
        stride_samples = self._stride_samples(fs_val)

        if self.padding == "same":
            x_pad = self._pad_same_1d(x, kernel_len=kernel.size(-1), stride=stride_samples, dilation=1)
            out = F.conv1d(x_pad, kernel, padding=0, stride=stride_samples)
        elif self.padding == "valid" or self.padding == 0:
            out = F.conv1d(x, kernel, padding=0, stride=stride_samples)
        else:
            out = F.conv1d(x, kernel, padding=int(self.padding), stride=stride_samples)

        dt = 1.0 / fs_val
        fs_out = fs_val / stride_samples
        return out * dt, fs_out
