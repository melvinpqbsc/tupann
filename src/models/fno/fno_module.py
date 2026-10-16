"""
Fourier Neural Operator (FNO) for 2D fields.

Based on Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations" (ICLR 2021).
The input lags are stacked along the channel dimension and mapped to all output lead times at once.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """Linear transform applied to the lowest Fourier modes, followed by an inverse FFT."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        # complex weights stored as real tensors (last dim = real/imag) so DDP and optimizers handle them natively
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, 2))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, 2))

    @staticmethod
    def compl_mul2d(x, weights):
        # (B, Cin, X, Y), (Cin, Cout, X, Y) -> (B, Cout, X, Y)
        return torch.einsum("bixy,ioxy->boxy", x, torch.view_as_complex(weights))

    def forward(self, x):
        B, _, H, W = x.shape
        m1 = min(self.modes1, H // 2)
        m2 = min(self.modes2, W // 2 + 1)
        x_ft = torch.fft.rfft2(x.float(), norm="ortho")

        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])

        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho").to(x.dtype)


class FNOBlock(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int, activation: bool = True):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.pointwise = nn.Conv2d(width, width, 1)
        self.activation = activation

    def forward(self, x):
        x = self.spectral(x) + self.pointwise(x)
        if self.activation:
            x = F.gelu(x)
        return x


class FNO2d(nn.Module):
    """
    Args:
        in_channels: number of input channels (stacked input lags).
        out_channels: number of output channels (stacked lead times).
        modes1, modes2: number of Fourier modes kept along H and W.
        width: hidden channel dimension.
        n_layers: number of Fourier layers.
        padding: zero padding (pixels) added on the bottom/right, since precipitation fields are not periodic.
        use_grid: if True, append normalized (x, y) coordinates as extra input channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int = 16,
        modes2: int = 16,
        width: int = 64,
        n_layers: int = 4,
        padding: int = 8,
        use_grid: bool = True,
        projection_channels: int = 128,
    ):
        super().__init__()
        self.padding = padding
        self.use_grid = use_grid

        self.lift = nn.Conv2d(in_channels + (2 if use_grid else 0), width, 1)
        self.blocks = nn.ModuleList(
            [FNOBlock(width, modes1, modes2, activation=(i < n_layers - 1)) for i in range(n_layers)]
        )
        self.proj1 = nn.Conv2d(width, projection_channels, 1)
        self.proj2 = nn.Conv2d(projection_channels, out_channels, 1)

    @staticmethod
    def get_grid(x):
        B, _, H, W = x.shape
        gy = torch.linspace(0, 1, H, device=x.device, dtype=x.dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        gx = torch.linspace(0, 1, W, device=x.device, dtype=x.dtype).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([gy, gx], dim=1)

    def forward(self, x):
        # x: (B, C_in, H, W)
        H, W = x.shape[-2:]
        if self.use_grid:
            x = torch.cat([x, self.get_grid(x)], dim=1)
        x = self.lift(x)
        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])
        for block in self.blocks:
            x = block(x)
        x = x[..., :H, :W]
        x = self.proj2(F.gelu(self.proj1(x)))
        return x
