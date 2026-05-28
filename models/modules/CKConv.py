import torch
import torch.nn as nn
import sys
import os

# Clond the ckconv repository from https://github.com/dwromero/ckconv and place it in the models/modules/ckconv folder.

# Check if ckconv is importable, if not, add it to path
try:
    import ckconv
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.dirname(current_dir) # models
    ckconv_root = os.path.join(models_dir, 'ckconv')
    if os.path.isdir(ckconv_root) and ckconv_root not in sys.path:
        sys.path.append(ckconv_root)
    try:
        import ckconv
    except ImportError:
        pass # Will likely fail downstream if still not found

# Import the reference implementation
# We access it via ckconv.nn.ckconv if ckconv package is resolved
try:
    from ckconv.ckconv.nn.ckconv import CKConv as RefCKConv
except ImportError:
    # Fallback to direct import from models if possible (if sys.path hack didn't work but we are in root)
    # But usually the sys.path append above handles 'ckconv'
    from models.modules.ckconv.ckconv.nn.ckconv import CKConv as RefCKConv


class CKConv(nn.Module):
    """
    Continuous Kernel Convolution (CKConv) wrapper compatible with SplineKernelConv1d API.
    
    This class wraps the `CKConv` implementation from `models/ckconv`, which uses a 
    Neural Network (KernelNet) to parameterize the convolution kernel. This allows for:
    1.  Resolution independence: The kernel is defined continuously and sampled at the current `fs`.
    2.  Long-range dependencies: Can handle very long kernels efficiently.
    
    This wrapper maintains the `SplineKernelConv1d` interface, specifically support for:
    - `kernel_duration` (seconds): Determines the effective field of view.
    - `stride` (seconds): Allows downsampling in the temporal domain.
    - `padding="same"`: Ensures output length alignment.
    """
    def __init__(
        self, 
        in_channels, 
        out_channels,
        kernel_duration=0.15,   # seconds, acts as the "receptive field" in time
        padding="same",         # "same" pads to maintain length, "valid" or int for standard
        stride=None,            # Stride in seconds; None => 1 sample stride (no downsampling)
        # CKConv specific optional arguments
        hidden_channels=32,     # Hidden size of the KernelNet (MLP)
        activation_function="Sine", # SIREN (Sine) is standard for implicit neural representations
        norm_type="",           
        omega_0=30.0,           # Frequency factor for SIREN
        weight_dropout=0.0,
        bias=True
    ):
                 
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_duration = float(kernel_duration)
        self.padding = padding
        # Stride is defined in seconds, converted to samples based on fs in forward()
        self.stride = None if stride is None else float(stride)
        
        # Instantiate the reference CKConv layer from the submodule.
        # We only use its `KernelNet` (implicit function) and `bias` parameter 
        # but re-implement the forward pass to support `kernel_duration` and `stride` dynamically.
        self.ckconv_layer = RefCKConv(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            activation_function=activation_function,
            norm_type=norm_type,
            dim_linear=1, # 1D Convolution
            bias=bias,
            omega_0=omega_0,
            weight_dropout=weight_dropout
        )
        
    def _stride_samples(self, fs_val: float) -> int:
        """Calculates stride in samples based on stride in seconds and sampling rate."""
        # Default: 1 sample if stride not provided
        if self.stride is None:
            return 1

        # Calculate samples: stride_sec * samples_per_sec
        s = int(round(self.stride * fs_val))
        return max(s, 1)

    @staticmethod
    def _pad_same_1d(x: torch.Tensor, kernel_len: int, stride: int, dilation: int = 1):
        """
        Pads input 'x' such that output length is ceil(InputLength / Stride).
        Matches TensorFlow 'SAME' padding logic.
        """
        L = x.size(-1)
        out_len = (L + stride - 1) // stride  # ceil(L/stride)
        effective_k = dilation * (kernel_len - 1) + 1
        pad_total = max((out_len - 1) * stride + effective_k - L, 0)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        if pad_total > 0:
            x = nn.functional.pad(x, (pad_left, pad_right))
        return x

    def forward(self, x, fs):
        """
        Forward pass with dynamic continuous kernel sampling.
        
        Args:
            x: Input tensor of shape (B, C_in, T)
            fs: Sampling rate (scalar or tensor). 
                Used to determine kernel length in samples and integration scaling.
            
        Returns:
            out: Convolved output (B, C_out, T_out)
            fs_out: Output sampling rate after striding
        """
        # Normalize fs to scalar for the batch (assumes consistent fs across batch)
        if isinstance(fs, torch.Tensor):
            fs_flat = fs.view(-1)
            if fs_flat.numel() > 0:
                fs_val = float(fs_flat[0].item())
            else:
                fs_val = 1.0 
        else:
            fs_val = float(fs)

        # -----------------------------------------------------------
        # 1. Determine Kernel Length
        # -----------------------------------------------------------
        # Calculate how many samples cover 'kernel_duration' seconds.
        # This makes the receptive field time-invariant regardless of fs.
        kernel_len = int(round(self.kernel_duration * fs_val))
        kernel_len = max(kernel_len, 1)
        
        # -----------------------------------------------------------
        # 2. Sample the Continuous Kernel
        # -----------------------------------------------------------
        # CKConv's KernelNet maps relative positions [-1, 1] to kernel weights.
        # We generate a grid of positions for the calculated length.
        rel_positions = torch.linspace(-1.0, 1.0, kernel_len).to(x.device).view(1, 1, kernel_len)
        
        # Query the implicit neural network to get weights for these positions.
        # Output shape: (1, Out*In, Len) -> View as (Out, In, Len)
        conv_kernel = self.ckconv_layer.Kernel(rel_positions)
        conv_kernel = conv_kernel.view(self.out_channels, self.in_channels, kernel_len)
        
        # -----------------------------------------------------------
        # 3. Perform Convolution
        # -----------------------------------------------------------
        stride_samples = self._stride_samples(fs_val)
        
        if self.padding == "same":
            # Manually pad to ensure 'same' output size behavior
            x_pad = self._pad_same_1d(x, kernel_len, stride=stride_samples, dilation=1)
            # Use 0 padding in conv1d since we already padded x
            out = nn.functional.conv1d(x_pad, conv_kernel, bias=self.ckconv_layer.bias, stride=stride_samples, padding=0)
            
        elif self.padding == "valid" or self.padding == 0:
            out = nn.functional.conv1d(x, conv_kernel, bias=self.ckconv_layer.bias, stride=stride_samples, padding=0)
            
        else:
            # Fixed integer padding
            out = nn.functional.conv1d(x, conv_kernel, bias=self.ckconv_layer.bias, stride=stride_samples, padding=int(self.padding))
        
        # -----------------------------------------------------------
        # 4. Integration Scaling
        # -----------------------------------------------------------
        # Continuous convolution is an integral: y(t) = integral(x(t-tau) * k(tau) dtau)
        # Discrete approx: sum(x * k) * dt.
        # dt = 1/fs.
        dt = 1.0 / fs_val
        out = out * dt
        
        fs_out = fs_val / stride_samples
        
        return out, fs_out
